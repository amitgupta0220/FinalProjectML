Dataset:
Dataset is taken from kaggle : https://www.kaggle.com/datasets/dongeorge/seed-from-uci

The project has a dependency of :
- numpy
- sklearn for confusion matrix
- pandas
- seaborn
- matplotlib

This project has following files:
1.) elbow_method.py
- To run this file simply run the command `python elbow_method.py`
- It will display a graph
- It simply gives an idea about what should be an optimal number of clusters for the dataset

2.) main.py
- To run this file simply run the command `python main.py`
- It is the core file and once executed it will run the algorithm over the data and display the accuracy of the model

3.) main_withPCA.py
- To run this file simply run the command `python main_withPCA.py`
- It is the core file and once executed it will run the algorithm over the data and display the accuracy of the model

4.) Seed_Data.csv
- It is the dataset which is being used by all files

5.) study_data.py
- To run this file simply run the command `python study_data.py`
- It simply reads the dataset and give the details about the data along with a graph

All other details are discussed in `report.docx` file.