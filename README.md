# Oracle
Oracle is a unified framework for normalization of buildings metadata. Different frameworks can be unified into a workflow and/or compared with each other in Oracle.


# Speficiation

## <a name="data_format"></a>Data Format

### Raw Metadata

1. Every BMS point is associated with a unique source identifier (srcid).
2. All BMS metadata is in the form of a table. A BMS point may have metadata in multiple columns.
    1. Example columns: Vendor-given names, BACNet names, BACNet unit.
    2. 

### Raw Timeseries Data
1. Every BMS point may produce a timeseries data associated with the corresponding srcid.
2. Its data format is **TODO**.

### Output Metadata in Brick
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

1. Each framework should be instantiated to [the common interface](https://github.com/jbkoh/oracle/blob/master/Oracle/frameworks/framework_interface.py).


## Workflow

1. Each framework aligned to the interface (``./Oracle/frameworks/framework_interface.py``) can be a part (i.e., Block) of a workflow to Brickify a building.
2. Workflow/Block interface is defined under **TODO**.
3. Workflow usage scenario:
    1. Each part is initiated with the raw data for a target building in the format described in [Data Format](data_format).
    1. In each iteration, each part runs algorithm sequentially.
    2. In each part, it receieves raw

## Benchmark


# Examples
