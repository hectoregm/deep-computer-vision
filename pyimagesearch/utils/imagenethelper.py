import numpy as np
import os

class ImageNetHelper:
    def __init__(self, config):
        self.config = config

        self.labelMappings = self.buildClassLabels()
        self.valBlacklist = self.buildBlacklist()

    def buildClassLabels(self):
        rows = open(self.config.WORD_IDS).read().strip.split("\n")
        labelMappings = {}

        for row in rows:
            (wordID, label, hrLabel) = row.split(" ")

            labelMappings[wordID] = int(label) - 1

        return labelMappings