This is a readme file for the RC Column Model process

(1) Run the get_data_peer.py file to download and process the data from the Peer Performance Database.
    This will create 416 files in the test_data folder.
    "test_001.json" through "test_416.json"
    Note: check that this code downloads the proper data.

(2) Run the create_data_matrix.py code to compute the non-dimensional parameters for each column. This will generate two files:
    i. data_spiral_wnd.csv
    ii. data_rect_wnd.csv

    The result will be two files:
    i. "data_spiral_wnd.csv"    This file contains both, the raw data for each of the spiral colums, and 6 additional values, corresponding 
                                to the non-dimensional parameters. It will also save the name of the test, to keep track of it.
    ii. "data_rect_wnd.csv"     This file contains both the raw data for each og the rectangular columns, and 6 additional non-dimensional paramers.
                                It will also save the name of the tests, to keep track of them.

    In the previous files, the id value is an unique id, that can be used to track the corresponding test throught the workflow.

(3) 