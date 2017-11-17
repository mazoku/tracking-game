# import the necessary packages
import commentjson as json


class Conf:
    def __init__(self, conf_path):
        # load and store the configuration and update the object's dictionary
        conf = json.loads(open(conf_path).read())
        self.__dict__.update(conf)

    def __getitem__(self, k):
        # return the value associated with the supplied key
        return self.__dict__.get(k, None)
