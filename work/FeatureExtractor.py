# -*- coding: utf-8 -*-
"""

Development package for a "Feature Extractor" class

"""
#
# Starting from a Surface described by a collection of "signal patterns" as a
# function of 2 variables (say a waterfall of functions of 1 variable, with a
# parametric dependence which defines the 2nd variable), the aim of this
# package is to facilitate the "highlighting" and the extraction of "peculiar"
# or "significant" patterns, which can be analysed further.
#
# The algorithm is based on a convolution integral of the surface under
# analysis with a symmetric Gaussian probe of defined variance (Sigma).
#
# As a consequence, broadly speaking, very localised features (spikes, sharp/
# short ridges/valleys) will be *dumped* while clusters of features will be
# enhanced and highlighted.   The [Max, Min] range will also be squeezed, with
# benefits for visualisation tools/commands.
#
# The process can be generalised to higher dimensions, although it's likely
# that going beyond 3D (i.e. a volume-like function) will require to abandon
# the convolution integral over a regula grid and adopt a Monte-Carlo approach.
#

import numpy as np


class FeatrXtrct:

    """ Feature Extractor Class """

    def __init__(self, surfData, prbWidth=11, prbSigma=0.5):
        """ feature highlights distribution.
        #
        Parameters:
            - surfData              = surface data : 2D [nRows, nColumns]
            - prbWidth (optional)   = edge length of symmetric Gaussian probe
                                      (odd number, so that max is at centre)
            - prbSigma (optional)   = sigma of the symmetric Gaussian probe
                                      exp(-(x^2+y^2)/sigma^2)
        """
        # get the input data
        #
        # define the probe
        hlfSteps = prbWidth/2
        if (hlfSteps.is_integer()):
            raise ValueError("Please choose an Odd Number as Probe Width")
        #
        self.hlfSteps = int(hlfSteps)
        nSteps = int(prbWidth)
        self.nSteps = nSteps
        xx = np.linspace(-1, 1, nSteps)
        funxx = np.exp(-xx*xx/prbSigma**2)
        self.prb = np.kron(funxx, funxx).reshape(nSteps, nSteps)
        #
        # done
        self.testData = surfData
        #

    def cnvWghtArray(self):
        """ Convolution Weight Array
            calculate weight matrix to alleviate any edge aberration by
            balancing results along edge strips and corners
        """
        hlfSteps = self.hlfSteps
        prb = self.prb
        dataSize = self.testData.shape
        maxRow = dataSize[0]
        maxCol = dataSize[1]
        #
        # mid lenghts
        midRow = int(np.round(maxRow/2 + 0.2))
        midCol = int(np.round(maxCol/2 + 0.2))
        #
        # Weight Array
        # ------------------------------------------------------------------------
        # initialise
        wghM = np.ndarray([maxRow, maxCol])
        wghM.fill(0.0)
        #
        # start filling sectors: make use of data symmetry
        #
        # NOTE: [0, 0] --> assuming top left origin - if different change reference
        #                  accordingly
        #
        # LH Top corner - range defined by the square
        # [0,0]-[0,hlfSteps-1]-[hlfSteps-1,hlfSteps-1]-[hlfSteps-1,0]
        for nR in range(0, hlfSteps):
            for nC in range(0, hlfSteps):
                wghM[nR, nC] = 1.0/sum(sum(prb[hlfSteps-nR:, hlfSteps-nC:]))
        #
        # LH - Top-Half strip - from LH Top Corner -> DOWN
        for nR in range(hlfSteps, midRow):
            for nC in range(0, hlfSteps):
                wghM[nR, nC] = 1.0/sum(sum(prb[:, hlfSteps-nC:]))
        #
        # Top - Left-Half Strip - from LH Corner -> RIGHT
        for nR in range(0, hlfSteps):
            for nC in range(hlfSteps, midCol):
                wghM[nR, nC] = 1.0/sum(sum(prb[hlfSteps-nR:, :]))
        #
        # Use symmetry to fill remaining corners and strips
        # Then bulk-filling of the core points with full probe integral
        #
        # --------------------------------------------------------------------------
        # corners
        # --------------------------------------------------------------------------
        #
        # LH - Bottom Corner
        wghM[maxRow-1:maxRow-(hlfSteps+1):-1, :hlfSteps] = \
            wghM[:hlfSteps, :hlfSteps].copy()
        #
        # RH - Top Corner
        wghM[:hlfSteps, maxCol-1:maxCol-(hlfSteps+1):-1] = \
            wghM[:hlfSteps, :hlfSteps].copy()
        #
        # RH - Bottom Corner
        wghM[maxRow-1:maxRow-(hlfSteps+1):-1, maxCol-1:maxCol-(hlfSteps+1):-1] =\
            wghM[:hlfSteps, :hlfSteps].copy()
        #
        # --------------------------------------------------------------------------
        # Strips
        # --------------------------------------------------------------------------
        # Complete Top Strip and get Bottom one by symmetry
        #
        # Top Right-Half Strip
        wghM[:hlfSteps, maxCol-(hlfSteps+1):midCol-1:-1] = \
            wghM[:hlfSteps, hlfSteps:midCol-1].copy()
        #
        #
        # Bottom Strip : maxRow-1 -> maxRow-(hlfStep+1), hlfSteps:maxCol-(hlfStep1+1)
        wghM[maxRow-1:maxRow-(hlfSteps+1):-1, hlfSteps:maxCol-hlfSteps] =\
            wghM[:hlfSteps, hlfSteps:maxCol-hlfSteps].copy()
        #
        #
        # --------------------------------------------------------------------------
        # Complete LH Strip and get RH one by symmetry
        #
        # LH Bottom-Half Strip
        wghM[maxRow-(hlfSteps+1):midRow-1:-1, :hlfSteps] = \
            wghM[hlfSteps:midRow-1, :hlfSteps].copy()
        #
        #
        # RH Strip : hlfSteps:maxRow-(hlfSteps+1), maxCol-1 -> maxCol-(hlfSteps+1)
        wghM[hlfSteps:maxRow-hlfSteps, maxCol-1:maxCol-(hlfSteps+1):-1] = \
            wghM[hlfSteps:maxRow-hlfSteps, :hlfSteps].copy()
        #
        # --------------------------------------------------------------------------
        #
        # remaining points
        commWgh = 1.0/sum(sum(prb))
        #
        wghM[hlfSteps:maxRow-hlfSteps, hlfSteps:maxCol-hlfSteps].fill(commWgh)
        #
        # -----------------------------------------------------------------------
        # Done Weight Matrix
        # -----------------------------------------------------------------------
        #
        return wghM

    def cnvInteg(self):
        """ Convolution Integral of Probe over Surface Data
            Given the probe definition, calculate the convolution integral of
            the probe surface on the data surface.

            Output:     cnvIntegReslt - result array, same shape as surfData
        """

        #
        # ---------------------------------------------------------------
        hlfSteps = self.hlfSteps
        prb = self.prb
        testData = self.testData
        dataSize = self.testData.shape
        maxRow = dataSize[0]
        maxCol = dataSize[1]
        #
        # get the weight array
        wghM = self.cnvWghtArray()
        #
        # initialise the results
        cnvIntegReslt = np.ndarray([maxRow, maxCol])
        cnvIntegReslt.fill(0.0)
        #
        # ---------------------------------------------------------------
        #
        # (1) probe distribution symmetric around the centre
        # (2) probe distribution centre coinciding with sample point at ( row, col )
        #
        # BULK
        # - row and col indexes > hlfSteps or < (maxRow - (hlfSteps+1))
        # => convolution with full probe distribution
        #
        # Corners
        # - Both indexes < hlfSteps+1 or > (maxRow-hlfSteps)
        # => convolution start from 1/4 of probe distribution
        #
        # Strips
        # - One of the indexes is either < hlfSteps+1 or > (maxRow-hlfSteps)
        # => convolution starts from 1/2 of probe distribution
        #
        # ---------------------------------------------------------------
        #
        # Corners
        # LH Top corner - range defined by the square
        # [0,0]-[0,hlfSteps-1]-[hlfSteps-1,hlfSteps-1]-[hlfSteps-1,0]
        #
        for nR in range(0, hlfSteps):
            for nC in range(0, hlfSteps):
                #
                # Top LH
                cnvIntegReslt[nR, nC] = wghM[nR, nC] * \
                    sum(sum(testData[:nR+hlfSteps+1, :nC+hlfSteps+1] *
                            prb[hlfSteps-nR:, hlfSteps-nC:]))
                #
                # Top RH
                cnvIntegReslt[nR, (maxCol-1)-nC] = wghM[nR, (maxCol-1)-nC] *\
                    sum(sum(testData[:nR+hlfSteps+1,
                                     (maxCol-1):(maxCol-1)-nC-(hlfSteps+1):-1] *
                            prb[hlfSteps-nR:, hlfSteps+nC::-1]))
                #
                # Bottom LH
                cnvIntegReslt[(maxRow-1)-nR, nC] = wghM[(maxRow-1)-nR, nC] *\
                    sum(sum(testData[(maxRow-1):(maxRow-1)-nR-(hlfSteps+1):-1,
                                     :nC+hlfSteps+1] *
                            prb[hlfSteps+nR::-1, hlfSteps-nC:]))
                #
                # Bottom RH
                cnvIntegReslt[(maxRow-1)-nR, (maxCol-1)-nC] = \
                    wghM[(maxRow-1)-nR, (maxCol-1)-nC] *\
                    sum(sum(testData[(maxRow-1):(maxRow-1)-nR-(hlfSteps+1):-1,
                                     (maxCol-1):(maxCol-1)-nC-(hlfSteps+1):-1] *
                            prb[hlfSteps+nR::-1, hlfSteps+nC::-1]))
        #
        # ---------------------------------------------------------------
        #
        # Strips
        #
        # Top - Bottom
        for nR in range(0, hlfSteps):
            for nC in range(hlfSteps, maxCol-(hlfSteps+1)):
                cnvIntegReslt[nR, nC] = wghM[nR, nC] * \
                    sum(sum(testData[:nR+hlfSteps+1, nC-hlfSteps:nC+hlfSteps+1] *
                            prb[:nR+hlfSteps+1, :]))
                #
                #
                cnvIntegReslt[(maxRow-1)-nR, nC] = wghM[(maxRow-1)-nR, nC] * \
                    sum(sum(testData[(maxRow-1):(maxRow-1)-(hlfSteps+1)-nR:-1,
                                     nC-hlfSteps:nC+hlfSteps+1] *
                            prb[hlfSteps+nR::-1, :]))
        #
        # LH- RH
        for nR in range(hlfSteps, maxRow-(hlfSteps)):
            for nC in range(0, hlfSteps):
                cnvIntegReslt[nR, nC] = wghM[nR, nC] * \
                    sum(sum(testData[nR-hlfSteps:nR+hlfSteps+1, :nC+hlfSteps+1] *
                            prb[:, :nC+hlfSteps+1]))
                #
                #
                cnvIntegReslt[nR, (maxCol-1)-nC] = wghM[nR, (maxCol-1)-nC] * \
                    sum(sum(testData[nR-hlfSteps:nR+hlfSteps+1,
                                     (maxCol-1):(maxCol-1)-(hlfSteps+1)-nC:-1] *
                            prb[:, hlfSteps+nC::-1]))
        #
        #
        # Bulk
        for nR in range(hlfSteps, maxRow-hlfSteps):
            for nC in range(hlfSteps, maxCol-hlfSteps):
                cnvIntegReslt[nR, nC] = wghM[nR, nC] * \
                    sum(sum(testData[nR-hlfSteps:nR+hlfSteps+1,
                                     nC-hlfSteps:nC+hlfSteps+1] * prb))
        #
        # Done
        return cnvIntegReslt
