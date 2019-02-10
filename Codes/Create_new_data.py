import pandas as pd
import pickle
import numpy as np

print("Dataset creation\n")

clinical_data = pd.read_excel('prad_tcga_clinical_data.xlsx')

#first 5 rows deducted because these patients are not present on the other dataset
clinical_data = clinical_data.iloc[5:] 
#clinical_data = clinical_data.head(494)

#Take gleason_score so that we can add it in the new dataset {five classes: 6, 7, 8, 9, 10}
gleason_score = clinical_data.loc[: , "GLEASON_SCORE"] 

gene_data = pd.read_excel('prad_tcga_genes.xlsx')

#dump in a pickle file for faster access
pickle.dump(gene_data, open('gene_exp.pickle', 'wb')) 

new_dataset = []
with open('gene_exp.pickle','rb') as f:
    df2 = pickle.load(f)
    columns = list(df2[df2.columns[0]])
    
for i in range(1,len(df2.columns)):
    #each column from prad_tcga_genes.xlsx data, 60483 columns in total for 60483 genes
    #Each row in the prad_tcga_genes.xlsx is now is stacked as columns in new_dataset
    new_dataset.append(df2.iloc[:,i]) 

#patients_ID column of clinical_data is now the index of the new data    
index = list(clinical_data[clinical_data.columns[0]]) 

new_dataset = np.array(new_dataset)

new_df = pd.DataFrame(data =new_dataset, index = index, columns = columns)

#Add gleason_score as another column in new_dataset file
new_df['gleason_score'] = np.array(gleason_score)

new_df.to_csv('Predict_gleasonScore.csv', sep=',')

