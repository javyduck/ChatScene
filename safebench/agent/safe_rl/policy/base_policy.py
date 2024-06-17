'''
 
Email:
Date: 2023-01-30 22:30:20
LastEditTime: 2023-02-26 00:48:48
Description: 
'''

from abc import ABC, abstractclassmethod


class Policy(ABC):
    def __init__(self) -> None:
        super().__init__()

    def _sanity_check(self, keys, config):
        '''
        Inline method, check if the config file contain certain key params.
        '''
        for key in keys:
            assert key in config.keys(), "missing %s parameter in the config!" % key

    def _config_to_attr(self, config):
        '''
        Inline method, convert dictionary config to class attributes
        '''
        for key in config.keys():
            self.__setattr__(key, config[key])

    @abstractclassmethod
    def learn_on_batch(self, batch: dict):
        '''
        Given a batch of data, train the policy
        '''
        raise NotImplementedError

    @abstractclassmethod
    def act(self, state):
        '''
        Given a single state, return the action, value, logp.
        This API is used to interact with the env.
        '''
        raise NotImplementedError

    def export_model(self, dir, name):
        '''
        Save the model to dir
        '''
        raise NotImplementedError

    def import_model(self, dir):
        '''
        Load the model from dir
        '''
        raise NotImplementedError