# Oracle
Oracle is a unified framework for normalization of buildings metadata. Different frameworks can be unified into a workflow and/or compared with each other in Oracle.


# Speficiation

## Data Format

### Raw Metadata

1. Every BMS point is associated with a unique source identifier (srcid).
2. All BMS metadata is in the form of a table. A BMS point may have metadata in multiple columns.
    1. Example columns: Vendor-given names, BACNet names, BACNet unit.
    2. 

### Output Metadata in Brick.
1. Result graph in Brick.
2. A map of confidence of triples.
    1. A key is a triple in string and the value is its confidence. If the triple is given by the user, it should be 1.0. E.g.,
        ```python
       {
         ("ex:znt", "rdf:type", "brick:Zone_Temperature_Sensor"): 0.9,
         ..
       }
       ```

## Framework Interface

1. Each framework should be instantiated to [the common interface](https://github.com/jbkoh/oracle/blob/master/Oracle/frameworks/framework_interface.py)
