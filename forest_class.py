# -*- coding: utf-8 -*-
from parameters import *
import healpy
import numpy as np
from healpy import query_disc
# import cosmology


class quasar(object):
    """ That's the main object that we will work with
    it has the information of delta field as well as weights
    for each quasar.
    """
    def __init__(self,fib,pl,name,ra,dec,lenght):
        self.ra = ra
        self.fib = fib
        self.pl = pl
        self.name = name
        self.dec = dec
        self.lenght = lenght

        phi = ra
        theta = halfpi - dec
        pix = healpy.ang2pix(nside,theta,phi)
        self.pix = pix

        self.dw = np.zeros(lenght, dtype=np.float64)
        self.we = np.zeros(lenght, dtype=np.float64)
        self.dc = np.zeros(lenght, dtype=np.float64)

        # Unitary cartesian coordinates
        self.x = np.cos(ra)*np.cos(dec)
        self.y = np.sin(ra)*np.cos(dec)
        self.z = np.sin(dec) 

        # Cartesian coordinates for each pixel in the forest
        self.rx = np.zeros(lenght, dtype=np.float64)
        self.ry = np.zeros(lenght, dtype=np.float64)
        self.rz = np.zeros(lenght, dtype=np.float64)

        # LogLambda and its average
        self.lambda_average = 0
        self.delta_lambda = np.zeros(lenght, dtype=np.float64)
        self.omega_delta_lambda2 = 0
        self.omega = 0


    def fill_dw(self, de, loglam, project):
        """
        This function fills delta*w for the quasar. It substracts the
        mean and first moment of each forest unless "project" is set to
        false.
        """
        if project:
            deltaprom = np.average(de,weights=self.we)
            self.lambda_average = np.average(loglam,weights=self.we)
            self.delta_lambda = loglam - self.lambda_average
            self.omega_delta_lambda2 = np.sum(self.delta_lambda**2*self.we)
            Ldprom = np.average(self.delta_lambda*de, weights=self.we)
            LLprom = np.average(self.delta_lambda*self.delta_lambda, weights=self.we)
            if LLprom == 0:
                print("aqui hay un cero")
                print(self.delta_lambda)
                print(self.we)
            self.dw = self.we * (de - deltaprom - self.delta_lambda*Ldprom/LLprom)
            self.omega = np.sum(self.we)
        else:
            self.dw = de*self.we

    def dot_product(self, other):
        """
        This function computes the dot product of the
        normalized position of two quasars. cos(theta)
        """
        prod = self.x * other.x + self.y * other.y + self.z * other.z
        return prod

    def neighborhood(self, data, angmax):
        """ This function finds the neighbors of a given forest to be used for two or three point correlation function.
        It adds the requierement ra_neigh>ra_self in order to repeat pairs
        Parammeters:
        self     forest
                    The forest used to find its neighbors
        angmax  float
                    Maximum angle in radians to consider another forest a neighbor of self
        """

        mumin = np.cos(angmax)
        neig_pix = query_disc(nside, [self.x, self.y, self.z], angmax, inclusive=True)
        neig_pix = [p for p in neig_pix if p in data]

        neighs = []
        for pix2 in neig_pix:
            for forest2 in data[pix2]:
                mu = self.dot_product(forest2)
                if mu > mumin and self.name != forest2.name and self.ra < forest2.ra:
                    neighs.append(forest2)
        return neighs

    def neighborhood_names(self, data, angmax):
        """ This function finds the neighbors of a given forest to be used for two or three point correlation function.
        It adds the requierement ra_neigh>ra_self in order to repeat pairs
        Parammeters:
        self     forest
                    The forest used to find its neighbors
        angmax  float
                    Maximum angle in radians to consider another forest a neighbor of self
        """

        mumin = np.cos(angmax)
        neig_pix = query_disc(nside, [self.x, self.y, self.z], angmax, inclusive=True)
        neig_pix = [p for p in neig_pix if p in data]

        neigh_names = []
        neigh_pixels = []
        for pix2 in neig_pix:
            for forest2 in data[pix2]:
                mu = self.dot_product(forest2)
                if mu > mumin and self.name != forest2.name and self.ra < forest2.ra:
                    neigh_names.append(forest2.name)
                    neigh_pixels.append(pix2)
        return [neigh_names, neigh_pixels]
