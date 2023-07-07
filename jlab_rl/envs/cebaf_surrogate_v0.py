# Author: Kishansingh Rajput
# Script: CEBAF cavities digital twin
# Org: Thomas Jefferson National Accelerator Facility

import numpy as np
import pandas as pd
import math



class cavity():
    """


    """
    def __init__(self, data, cavity_id):
        """

        :param pathToCavityData:
        """
        
        self.cavity_id = cavity_id
        try:
            row = data[data["cavity_id"] == cavity_id]
        except(e):
            print(e)

        if len(row) < 1:
            print("Cavity-id ", self.cavity_id, " not found in the dataset...")
        
        self.length = float(row["length"])
        self.type = str(list(row['type'])[0])
        self.Q0 = float(row["Q0"])
        self.trip_slope = float(row['trip_slope'])
        self.trip_offset = float(row['trip_offset'])
        self.shunt = float(row['shunt_impedance'])
        self.max_gset = float(row['max_gset'])
        self.ops_gset_max = float(row['ops_gset_max'])
        if pd.isna(self.ops_gset_max):
            self.max_gset_to_use = self.max_gset
        else:
            self.max_gset_to_use = self.ops_gset_max
        self.min_gset = 3.0
        
        self.gradient = self.max_gset_to_use
        print("Type: ", str(self.type))
        if self.type in ["C75", "C100"]:
            self.min_gset = 5.0
            def computeHeat():
                return ((self.gradient**2) * self.length * 1e12) / (self.shunt * self.Q0)
        else:
            def computeHeat():
                return ((self.gradient**2) * self.length * 1e12) / (self.shunt * self.Q0)

        self.RFheat = computeHeat
        

    def describe(self):
        """

        :return:
        """
        print("Cavity type: ", self.cavity_id)
        print("Cavity current gradient: ", self.gradient)
        print("Cavity length: ", self.length)
        print("Cavity Q0: ", self.Q0)
        print("Cavity trip slope: ", self.trip_slope)
        print("Cavity trip offset: ", self.trip_offset)
        print("Cavity shunt: ", self.shunt)
        print("Cavity max_gset: ", self.max_gset)
        print("Cavity ops gset max: ", self.ops_gset_max)
        print("max gset to use: ", self.max_gset_to_use)



    def setGradient(self, grad):
        """

        :param gradArray:
        :return:
        """
        if type(grad) in [int, float, np.float32, np.float64]:
            if grad < self.min_gset:
                self.gradient = self.min_gset
            elif grad > self.max_gset_to_use:
                self.gradient = self.max_gset_to_use
            else:
                self.gradient = float(grad)
        else:
            print("Error: ", self.cavity_id, " gradient must be a float or integer and not ", type(grad))


    def getRFHeat(self):
        """

        :return:
        """
        return self.RFheat()


    def getTripRate(self):
        """

        :return:
        """
        if pd.isna(self.trip_slope) or pd.isna(self.trip_offset):
            return 0.0
        return math.exp(-10.268+self.trip_slope*(self.gradient - self.trip_offset))

    def getGradient(self):
        return self.gradient

    def getCavityState(self):
        if self.max_gset_to_use == self.min_gset:
            return 0.5
        else:
            return ((self.gradient - self.min_gset) / (self.max_gset_to_use - self.min_gset))


    def getEnergy(self):
        return self.length * self.gradient

    def reset(self):
#         gset = self.max_gset_to_use - 0.95  #North linac
        gset = (self.max_gset_to_use-3.)/2. + 3. #For 1L06 cryomodule subtracting 3 from the max gradient produces a gradient setting that is at the center of energy constraint
        if gset < self.min_gset:
            gset = self.min_gset
        self.gradient = gset
#         print("Length: ", self.length)
#         print("Reset Grad: ", self.gradient)


class digitalTwin():
    """

    """
    def __init__(self, path_cavity_data, linac="North"):
        """

        :param path_cavity_data:
        """
        data = pd.read_pickle(path_cavity_data)
        cavity_ids = data["cavity_id"]
        self.cavities = []
        self.cavity_order = []
        for i in range(len(cavity_ids)):
            if linac.lower() == "north" or linac.lower() == "n":
                if cavity_ids.iloc[i][0] == '1':
                    self.cavity_order.append(cavity_ids.iloc[i])
                    self.cavities.append(cavity(data, cavity_ids.iloc[i]))
            elif linac.lower() == "south" or linac.lower() == "s":
                if cavity_ids.iloc[i][0] == '2':
                    self.cavity_order.append(cavity_ids.iloc[i])
                    self.cavities.append(cavity(data, cavity_ids.iloc[i]))
            elif linac.lower() == "1l06_test1":
                if cavity_ids.iloc[i][0] == '1' and cavity_ids.iloc[i][2] == '0' and cavity_ids.iloc[i][3] == '6' and cavity_ids.iloc[i][5] in ['1']:
                    self.cavity_order.append(cavity_ids.iloc[i])
                    self.cavities.append(cavity(data, cavity_ids.iloc[i]))
            elif linac.lower() == "1l06_test2":
                if cavity_ids.iloc[i][0] == '1' and cavity_ids.iloc[i][2] == '0' and cavity_ids.iloc[i][3] == '6' and cavity_ids.iloc[i][5] in ['1', '2']:
                    self.cavity_order.append(cavity_ids.iloc[i])
                    self.cavities.append(cavity(data, cavity_ids.iloc[i]))
            elif linac.lower() == "1l06_test4":
                if cavity_ids.iloc[i][0] == '1' and cavity_ids.iloc[i][2] == '0' and cavity_ids.iloc[i][3] == '6' and cavity_ids.iloc[i][5] in ['1', '2', '3', '4']:
                    self.cavity_order.append(cavity_ids.iloc[i])
                    self.cavities.append(cavity(data, cavity_ids.iloc[i]))
            elif linac.lower() == "1l06_test8":
                if cavity_ids.iloc[i][0] == '1' and cavity_ids.iloc[i][2] == '0' and cavity_ids.iloc[i][3] == '6': # and cavity_ids.iloc[i][5] in ['1', '2']:
                    self.cavity_order.append(cavity_ids.iloc[i])
                    self.cavities.append(cavity(data, cavity_ids.iloc[i]))
            elif linac.lower() == "1l10_test1":
                if cavity_ids.iloc[i][0] == '1' and cavity_ids.iloc[i][2] == '1' and cavity_ids.iloc[i][3] == '0' and cavity_ids.iloc[i][5] in ['1']:
                    self.cavity_order.append(cavity_ids.iloc[i])
                    self.cavities.append(cavity(data, cavity_ids.iloc[i]))
            elif linac.lower() == "1l10_test2":
                if cavity_ids.iloc[i][0] == '1' and cavity_ids.iloc[i][2] == '1' and cavity_ids.iloc[i][3] == '0' and cavity_ids.iloc[i][5] in ['1', '2']:
                    self.cavity_order.append(cavity_ids.iloc[i])
                    self.cavities.append(cavity(data, cavity_ids.iloc[i]))
            elif linac.lower() == "1l10_test4":
                if cavity_ids.iloc[i][0] == '1' and cavity_ids.iloc[i][2] == '1' and cavity_ids.iloc[i][3] == '0' and cavity_ids.iloc[i][5] in ['1', '2', '3', '4']:
                    self.cavity_order.append(cavity_ids.iloc[i])
                    self.cavities.append(cavity(data, cavity_ids.iloc[i]))
            elif linac.lower() == "1l10_test8":
                if cavity_ids.iloc[i][0] == '1' and cavity_ids.iloc[i][2] == '1' and cavity_ids.iloc[i][3] == '0': # and cavity_ids.iloc[i][5] in ['1', '2']:
                    self.cavity_order.append(cavity_ids.iloc[i])
                    self.cavities.append(cavity(data, cavity_ids.iloc[i]))
            else: #Injector
                if cavity_ids.iloc[i][0] == '0':
                    self.cavity_order.append(cavity_ids.iloc[i])
                    self.cavities.append(cavity(data, cavity_ids.iloc[i]))
        self.name = linac

        if linac.lower() in ["north", "n", "south", "s"]:
            self.energyConstraint = 1050
            self.energyMargin = 2
        elif linac.lower() == "1l06_test1":
            self.energyConstraint = 2.5 #7.85
            self.energyMargin = 0.1
        elif linac.lower() == "1l06_test2":
            self.energyConstraint = 7.85 #7.85
            self.energyMargin = 0.1
        elif linac.lower() == "1l06_test4":
            self.energyConstraint = 15.95 #7.85
            self.energyMargin = 0.2
        elif linac.lower() == "1l06_test8":
            self.energyConstraint = 31.8 #7.85
            self.energyMargin = 0.2
        elif linac.lower() == "1l10_test2":
            self.energyConstraint = 4.5 #7.85
            self.energyMargin = 0.1
        elif linac.lower() == "1l10_test4":
            self.energyConstraint = 9.1 #7.85
            self.energyMargin = 0.2
        elif linac.lower() == "1l10_test8":
            self.energyConstraint = 20.08#17.3 #7.85
            self.energyMargin = 0.2
        else:
            self.energyConstraint = 126.5
            self.energyMargin = 0.22
            
        self.min_energy, self.max_energy = self.getEnergyGainBoundaries()
        self.min_heat, self.max_heat, self.min_trip_rate, self.max_trip_rate = self.getHeatAndTripRanges()
        print("Using min, max energy boundaries as: ", self.min_energy, self.max_energy)
        print("Max Gradients: ", self.getMaxGradients())
        print("Min Gradients: ", self.getMinGradients())
        
            
    def getEnergyGainBoundaries(self):
        e_max, e_min = 0.0, 0.0
        for cavity in self.cavities:
            e_min += cavity.length * cavity.min_gset
            e_max += cavity.length * cavity.max_gset_to_use
        return e_min, e_max
    
    
    def getHeatAndTripRanges(self):
        minHeat, maxHeat, minTrip, maxTrip = 0.0, 0.0, 0.0, 0.0
        for cavity in self.cavities:
            cavity.setGradient(cavity.min_gset)
            h1 = cavity.getRFHeat()
            T1 = cavity.getTripRate()
            cavity.setGradient(cavity.max_gset_to_use)
            h2 = cavity.getRFHeat()
            T2 = cavity.getTripRate()
            cavity.reset()
            minHeat += h1
            maxHeat += h2
            minTrip += T1
            maxTrip += T2

        return minHeat, maxHeat, 3600*minTrip, 3600*maxTrip

    def list_cavities(self):
        """

        :return:
        """
        return self.cavity_order

    def describeCavity(self, cavity_id):
        """

        :param cavity_id:
        :return:
        """
        indx = self.cavity_order.index(cavity_id)
        self.cavities[indx].describe()

    def setGradients(self, grad_array):
        """

        :param grad_array:
        :return:
        """
        for i in range(len(self.cavities)):
            self.cavities[i].setGradient(grad_array[i])

    def getGradients(self):
        """
        :return:
        """
        gradients = []
        for cavity in self.cavities:
            gradients.append(cavity.getGradient())
        return np.array(gradients)

    def getState(self):
        """

        :return:
        """
        state_vars = []
        for cavity in self.cavities:
            state_vars.append(cavity.getGradient())
        state_vars.append(self.getEnergyGain())
        return np.array(state_vars)

    def getRFHeat(self):
        """

        :return:
        """
        heat = 0.0
        for cavity in self.cavities:
            heat += cavity.getRFHeat()
        return heat

    def getTripRates(self):
        """

        :return:
        """
        tr = 0.0
        for cavity in self.cavities:
            tr += cavity.getTripRate()
        return 3600*tr

    def getEnergyGain(self):
        """

        :return:
        """
        e = 0.0
        for cavity in self.cavities:
            e += cavity.getEnergy()
        return e

    def getMinGradients(self):
        """

        """
        min_grads = []
        for cavity in self.cavities:
            min_grads.append(cavity.min_gset)
        return np.array(min_grads)

    def getMaxGradients(self):
        """

        """
        max_grads = []
        for cavity in self.cavities:
            max_grads.append(cavity.max_gset_to_use)
        return np.array(max_grads)

    def reset(self):
        for cavity in self.cavities:
            cavity.reset()
        state = self.getState()
        return state

    def getEnergyConstraint(self):
        return self.energyConstraint

    def getEnergyMargin(self):
        return self.energyMargin

    def updateGradients(self, delta):
        """

        """
        new_grads = self.getGradients() + delta
        self.setGradients(new_grads)