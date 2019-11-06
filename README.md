# Plastering
Plastering is a unified framework for normalization of buildings metadata. Different frameworks can be unified into a workflow and/or compared with each other in Plastering.

# Getting Started

## Installation
1. Install MongoDB: [instruction](https://docs.mongodb.com/manual/installation/#mongodb-community-edition-installation-tutorials)
2. Install Dependencies: `pip install -r requirements.txt`
3. Install Plastering package: `python setup.py install`
4. ~~Download dataset [here](https://drive.google.com/drive/u/0/folders/1I-hV6j7AQSm4Q_pd3tc9_tBEJUIKveQg). This link is not public yet. You may use synthesized data to test the algorithms for now.~~ Unfortunately, UCSD does not approve publicly sharing the data. We may have a procedure to sign an agreement, but it's still under development. Until then please refer to a synthesized data as specified in [an example](https://github.com/plastering/plastering/blob/refactor-inferencer/examples/tutorial/load_data.py).

## Example with synthesized data.
1. Load data: ``python examples/tutorial/load_data.py``
2. Run Zodiac: ``python examples/tutorial/zodiac_tutorial.py``
    - This will print out accuracy (F1 scores) step by step.

## Example with SDH data
1. Load ata: `python examples/tutorial/load_data_sdh.py`
2. Run Scrabble: `python examples/tutorial/scrabble_tutorial.py`
    - This produces `scrabble_output.ttl`.
    - There will be an update about how to produce other types of results (metrics, other files, etc.)

## Other examples
1. Run Zodiac test: ``python test_zodiac.py``
2. Run Workflow test: ``python test_workflow.py``
3. Run Zodiac experiments: ``python scripts/exp_zodiac.py ap_m``
4. Produce figures: ``python scripts/result_drawer.py``


# Speficiation

## <a name="data_format"></a>Data Format

### Raw Metadata
0. It is defined as ``RawMetadata`` inside ``plastering/metadata_interface.py``.
1. Every BMS point is associated with a unique source identifier (**srcid**) and a **building** name.
2. All BMS **metadata** is in the form of JSON document. A BMS point corresponds to a row with metadata possibly in multiple entries. Example:
    ```json
    {
        "srcid": "123-456",
        "VendorGivenName": "RM-101.ZNT",
        "BACnetName": "VMA101 Zone Temp",
        "BACnetUnit": 64
    }
    ```

### Ground Truth of Metadata (LabeledMetadata)
0. It is defined as ``LabeledMetadata`` inside ``plastering/metadata_interface.py``.
1. **tagsets**: Any TagSets associated with the point.
2. **point_tagset**: Point TagSet among the associated TagSets. If it's not defined, one may select Point-related TagSets from **tagsets**.
2. **fullparsing**: Each entry has parsing results. An example for ``123-456``'s VendorGivenName:
    1. Tokenization: ``["RM", "-", "101", ".", "ZN", "T"]``
    2. Token Labels: ``["Room", None, "leftidentifier", None, "Zone", "Temperature"]``
    3. (Though Plastering by default supports the above token-label sets, different tokenization rules may apply from a framework. For example, one may want to use ``ZNT -> Zone_Temperature_Sensor`` instead. Such combinations can be extended later.)
3. One may use a part of different label types or add a new label type if needed.

### Raw Timeseries Data
1. Every BMS point may produce a timeseries data associated with the corresponding srcid.
2. Its data format is **TODO**.

### Output Metadata in Brick
1. **Brick graph**: Result graph in Brick (in Turtle syntax).
    ```turtle
    ex:RM_101_ZNT rdf:type brick:Zone_Temperature_Sensor .
    ex:RM_101 rdf:type brick:Room .
    ex:RM_101_ZNT bf:hasLocation ex:RM_101 .
    ```
2. **Confidences**: A map of confidence of triples.
    1. A key is a triple in string and the value is its confidence. If the triple is given by the user, it should be 1.0. E.g.,
        ```python
       {
         ("ex:RM_101_ZNT", "rdf:type", "brick:Zone_Temperature_Sensor"): 0.9,
         ..
       }
       ```

## Framework Interface
1. Each framework should be instantiated as [the common interface](https://github.com/jbkoh/plastering/blob/master/plastering/inferencers/inferencer.py).

### Common Procedure
1. Prepare the data in MongoDB. Example: ``data_init.py``
2. The number of seed samples are given and a framework is initialized with the number as well as the other configurations, which depend on the framework as different framework may require different initial inputs.  
    ```python
    conf = {
            'source_buildings': ['ebu3b'],
            'source_samples_list': [5],
            'logger_postfix': 'test1',
            'seed_num': 5}
    target_building = 'ap_m'
    scrabble = ScrabbleInterface(target_building, conf)
    ```
3. Start learning the entire building's metadata with the instance.  
    ```python
    scrabble.learn_auto() # This function name may change in the near future.
    ```
    Each step inside ``learn_auto()`` looks like this:
    1. Pick most informative samples in the target building.  
        ```python
        # this code is different from acutal Scrabble code as it internally contains all the process.
        new_srcids = self.scrabble.select_informative_samples(10)
        ``` 
    2. Update the model  
        ```python
        self.update_model(new_srcids)
        ```
    3. Infer with the update model
        ```python
        pred = self.scrabble.predict(self.target_srcids)
        ```
    4. Store the current model's performance. 
        ```python
        self.evaluate()
        ```

## Workflow
1. Each framework aligned to the interface (``./plastering/inferencers/inferencer.py``) can be a part, called *Inferencer*, of a workflow to Brickify a building.
2. Workflow/Inferencer interface is defined under **TODO**.
3. Workflow usage scenario:
    1. Each part is initiated with the raw data for target buildings in the format described in [Data Format](#data_format).
    2. In each iteration, each part runs algorithm sequentially.
    3. In each part, it receieves the result from the previous part and samples from an expert if necessary.

## Benchmark
1. Plastering also can be used to benchamrk different algorithms. It defines the common dataset and interactions with the expert providing learning samples.
2. Benchmark usage scenario.


# Examples

1. Initialize data 
    ```bash
    python data_init.py -b ap_m
    ```

2. Test with Zodiac
    ```bash
    python test_zodiac.py
    ```


