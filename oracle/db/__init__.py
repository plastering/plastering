from mongoengine import *

connect('oracle')


class RawMetadata(Document):
    srcid = StringField(required=True)
    building = StringField(required=True)
    metadata = DictField()

class LabeledMetadata(Document):
    srcid = StringField(required=True)
    building = StringField(required=True)
    fullparsing = DictField()
    tagsets = ListField(StringField())


from pymongo import MongoClient


class OracleDatabase(object):
    """docstring for OracleDatabase
    # "raw" data model
    {
        "srcid": "srcid1",
        "building": "ebu3b",
        "metadata": {
            "bacnet_name": "ZNT.RM-101",
            "vendor_given_name": "ZoneTemp VAV-230",
            "bacnet_unit": 75,
            "bacnet_point_type": "Analog Input"
        },
        "timeseries": BSON
    }
    # "labeled" data model
    {
        "srcid": "srcid1",
        "building": "ebu3b",
        "metadata_char_labels": {
            "bacnet_name": [("Z", "B_Zone")],
            ...
        },
        "metadata_tagset_labels": ["Zone_Temperature_Sensor", "Room"]
        "triples": [("ZNT.RM-101", RDF.type, "Zone_Temperature_Sensor"),
                    ("RM-101", RDF.type, "Room"),
                    ("ZNT_.RM-101", BF.hasLocation, "RM-101)],
    }

    """
    def __init__(self):
        self.cli = MongoClient('localhost', 27017)
        self.db = self.cli.get_database('oracle')
        self.raw = self.db.get_collection('raw')
        self.labeled = self.db.get_collection('labeled')

    def _form_srcid_query(self, srcid, building=None):
        q = {'srcid': srcid}
        if building:
            q['building'] = building
        return q

    def add_raw_metadata(self, srcid, tag, value, building=None):
        """
        srcid: "srcid1"
        tag: "bacnet_name"
        value: "ZNT.RM-101"
        """
        q = self._form_srcid_query(srcid, building)
        self.raw.update_one(q,
                            {'$set': {'metadata.' + tag: value}}, 
                            upsert=True)
    def add_fullparsing(self, srcid, tag, fullparsing, building=None):
        """
        srcid: "srcid1"
        tag: "bacnet_name"
        value: [("Z", "B_Zone"), ...]
        """
        q = self._form_srcid_query(srcid, building)
        self.labeled.update_one(q,
                            {'$set': {'fullaprsing.' + tag: fullparsing}},
                            upsert=True)
    def add_tagsets(self, srcid, tagsets):
        """
        srcid: "srcid1"
        tagsets: {"Zone_Temperature_Sensor", "Room"}
        """
        q = self._form_srcid_query(srcid, building)
        self.labeled.update_one(q,
                                 {'$set': {'tagsets': tagsets}},
                                 upsert=True)

    def get_all_srcids(self, building):
        return self.raw.distinct('srcid')

    def get_all_labeled_srcids(self, building):
        return self.labeled.distinct('srcid')
        
    
    def get_raw(self, srcid):
        pass

    def get_srcids(self, building):
        pass
    
    def get_raw():
        pass

