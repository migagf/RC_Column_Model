This is a readme file for the RC Column Model process

(1) Run the get_data_peer.py file to download and process the data from the Peer Performance Database.
    This will create 416 files in the test_data folder.
    "test_001.json" through "test_416.json"
    
    get_data_peer is a standalone function meaning that it does not call any functions other than the defined in there.

(2) Run the create_data_matrix.py code to compute the non-dimensional parameters for each column. Right now, it olny does the job for
    spiral columns. In the future, rectangular columns in the database will also be added.

    The result will be a file:
    "data_spiral_wnd.csv" which will contain both, the raw data for each of the spiral colums, and 6 additional values, corresponding 
    to the non-dimensional parameters.

(3) 