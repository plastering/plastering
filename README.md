# Oracle
Oracle is a unified framework for normalization of buildings metadata. Different frameworks can be unified into a workflow and/or compared with each other in Oracle.


# Speficiation

## <a name="data_format"></a>Data Format

### Raw Metadata
1. Every BMS point is associated with a unique source identifier (srcid).
2. All BMS metadata is in the form of a table in CSV. A BMS point corresponds to a row with metadata possibly in multiple columns. Example:

    | SourceIdentifier | VendorGivenName | BACNetName      | BACNetUnit |
    |------------------|-----------------|-----------------|------------|
    | 123-456          | RM-101.ZNT      | VAV101 ZoneTemp | 64         |

### Ground Truth of Metadata
1. Each row in the table has corresponding Brick triples. 
    1. E.g.,
        ```turtle
        ex:RM_101_ZNT rdf:type brick:Zone_Temperature_Sensor .
        ex:RM_101 rdf:type brick:Room .
        ex:RM_101_ZNT bf:hasLocation ex:RM_101 .
        ```  
    2. Framework interface has abstraction of interacting these triples.
2. Each cell has parsing results. An example for 123456.VendorGivenName:
    1. Tokenization: ``["RM", "-", "101", ".", "ZN", "T"]``
    2. Token Labels: ``["Room", None, "left-identifier", None, "Zone", "Temperature"]``  
    3. Though Oracle by default supports the above token-label sets, different tokenization rules may apply from a framework. For example, one may want to use ``ZNT -> Zone_Temperature_Sensor``. Such combinations can be extended later.
3. One may use a part of different label types or add a new label type if needed. 

### Raw Timeseries Data
1. Every BMS point may produce a timeseries data associated with the corresponding srcid.
2. Its data format is **TODO**.

### Output Metadata in Brick
1. Result graph in Brick (in Turtle syntax).
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
1. Each framework aligned to the interface (``./Oracle/frameworks/framework_interface.py``) can be a part, called *Block*, of a workflow to Brickify a building.
2. Workflow/Block interface is defined under **TODO**.
3. Workflow usage scenario:
    1. Each part is initiated with the raw data for target buildings in the format described in [Data Format](#data_format).
    2. In each iteration, each part runs algorithm sequentially.
    3. In each part, it receieves the result from the previous part and samples from an expert if necessary.

## Benchmark
1. Oracle also can be used to benchamrk different algorithms. It defines the common dataset and interactions with the expert providing learning samples.
2. Benchmark usage scenario.


# Examples
**TODO**
