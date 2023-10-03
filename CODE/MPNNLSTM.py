
#%% Libraries  
# Torch-related libraries 
import torch  
from torch_geometric.data import Data, DataLoader
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import MPNNLSTM

# Auxiliary libraries
from sklearn.preprocessing import StandardScaler

# Other libraries
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
from tqdm import tqdm
import time as TIME 

start_time = TIME.time() # To measure time
#%% Parameters
# Features's Selection
# Order: [0]ACTNOW_RISK    >> CLASSES: 0, 1, 2, 3, 4 
# [1]ACTNOW_CASES  [2]ACTNOW_DEATH  [3]ESTIM_RT    >> Features COVID-19
# [4]SAT_AOD  [5]SAT_TEMP  [6]SAT_RH    >> Features SATELLITE
#
# Training Parameters
percenTrain = 0.8   # Percentage for training: value in (0,1] 
NEpochs = 500 # Number of Epochs for training (ORIGINAL: 600)  
# Historical and Forecasting parameters
LBack = 4 # Historical: LBack>=0 , where 0 means only takes current X observation
LForward = 15 # Forecasting: LForward>=1 , where 0 means takes tomorrow's Y value
# Model and Training Parameters
#percen_embed = 1/3 # Percentage of dimension embeding
NTT = 10 # Number of RUNS (ORIGINAL: 10)  

# Period of time (Range)  
initialDay = '02/01/2020' 
finalDay = '12/31/2020'   
#********* Dataset Selection *********** <<< HERE !!! >>>


folderIndexes = 'Sel_0_1_2_4_5_6_7_8'
#pathInputALL = 'drive/MyDrive/Colab Notebooks/DATASET/'+letter2State+'/'+folderIndexes+'/'  
# Adjacent Matrix
nameFileCounties = 'IG_List_County.csv' 
nameFileADJ = 'IG_County_ADJ.csv' 
# Input Files
nameIN_RISK = 'NX_RISK_Covid19.csv'
nameIN_CASES = 'NX_CASES_Covid19.csv'
nameIN_DEATH = 'NX_DEATH_Covid19.csv'
nameIN_RT = 'NX_RT_Covid19.csv'
nameIN_AOD = 'NX_AOD_Satellite.csv'
nameIN_TEMP = 'NX_TEMP_Satellite.csv'
nameIN_RH = 'NX_RH_Satellite.csv'
nameIN_HOSPI = 'NX_HOSPI_Covid19.csv'
nameIN_ICU = 'NX_ICU_Covid19.csv'
nameIN_SEVERITY = 'NX_SEVERITY_Covid19.csv'



#**** Selected Target 
sel_TARGET = nameIN_HOSPI
STATE = "CA PA TX"
FEATURE = "+SOCIAL +AOD"
list_feature = FEATURE.split()
list_state = STATE.split()

sel_FEATURES1 = [nameIN_HOSPI] 
sel_FEATURES2 = [nameIN_HOSPI, nameIN_AOD]
sel_FEATURES3 = [nameIN_HOSPI, nameIN_TEMP]  
sel_FEATURES4 = [nameIN_HOSPI, nameIN_RH]

nameIN_SOCIO = 'NX_Socioeconomic.csv'
sel_FEATURES5 = [nameIN_HOSPI, nameIN_SOCIO]
list_FEATURE = (sel_FEATURES5,sel_FEATURES2)


for i in range(len(list_feature)):
  for letter2State in list_state:
        FEATURES = list_feature[i] 
        sel_FEATURES = list_FEATURE[i]
        pathInputALL = 'DATASET/'+letter2State+'/'+folderIndexes+'/'  
        pathOutput = 'OUTPUT/MPNNLSTM/'+letter2State+'/'+FEATURES+'/'

        print(pathOutput)
            #%% Functions
        def getDaily_Dataframe(dfData, initialDay, finalDay):
            # Starting index 
            indStart = dfData.index.get_loc(initialDay)
            # Ending index 
            indFinal = dfData.index.get_loc(finalDay)
            # Create new daily cases/deaths dataframe
            dfData_NewD = dfData.iloc[0:5].copy()
            for iD in range(indStart, indFinal+1):  
                cases_T = [float(a) for a in list(dfData.loc[dfData.index[iD]])]
                cases_T_DBack = [float(a) for a in list(dfData.loc[dfData.index[iD-1]])]
                arrayNewCases = np.subtract(cases_T, cases_T_DBack)
                arrayNewCases[arrayNewCases<0] = 0.0  # In case error in monitoring-data 
                df_Row_aux = pd.DataFrame([arrayNewCases.tolist()], columns=dfData.columns)
                # To append in the dataframe 
                dfData_NewD = dfData_NewD.append(df_Row_aux)
            # Change Index name 
            lisIDX_OUT = list(dfData.index[indStart:(indFinal+1)])
            lisIDX_OUT.insert(0, dfData.index[4])
            lisIDX_OUT.insert(0, dfData.index[3])
            lisIDX_OUT.insert(0, dfData.index[2])
            lisIDX_OUT.insert(0, dfData.index[1])
            lisIDX_OUT.insert(0, dfData.index[0])
            dfData_NewD.index = lisIDX_OUT
            dfData_NewD.index.name = dfData.index.name
            return dfData_NewD.copy()

        def replace_greater_Dataframe(dfData, maxVal):
            dfData_NewD = dfData.copy() 
            for strFIPS in dfData.columns:  
                lisStr = dfData_NewD[strFIPS].iloc[5:].values.tolist()
                arrayAux = np.array( [float(a) for a in lisStr] ) 
                if(np.sum(arrayAux<=maxVal)>0):
                    valReplace = np.quantile(arrayAux[arrayAux<=maxVal], 0.2)
                else:
                    valReplace = 0.05
                lisFloat = np.where(arrayAux>maxVal, valReplace, arrayAux).tolist()
                lisNewStr = [str(a) for a in lisFloat]
                dfData_NewD[strFIPS].iloc[5:] = lisNewStr
            # Return
            return dfData_NewD.copy()

        def getSelected_Period_Dataframe(dfData, initialDay, finalDay):
            dfData_NewD = pd.concat([ dfData.iloc[0:5].copy(), dfData[(dfData.index>=initialDay) & (dfData.index<=finalDay)].copy() ])  
            return dfData_NewD.copy()

        def get_Standarized_Dataframe(df_Target, ind_Before_Y):
            scaler_Target = StandardScaler()
            scaler_Target.fit(df_Target.iloc[5:ind_Before_Y])
            dataNorm_Target = scaler_Target.transform(df_Target.iloc[5:]) # Standarization 
            dfTarget_Pred = pd.DataFrame(dataNorm_Target, index = df_Target.index[5:], columns=df_Target.columns)  
            df_Target = pd.concat( [ df_Target.iloc[:5], dfTarget_Pred] ).copy()
            return [scaler_Target, df_Target]

        def save_Total_Statistics(ayy_train, txtIndex, colTitles, nameFile):
            ayy_train = np.concatenate((ayy_train, np.mean(ayy_train, axis=0).reshape(1,2)))  # Mean 
            ayy_train = np.concatenate((ayy_train, np.std(ayy_train, axis=0).reshape(1,2)))  # Std 
            dfResul_ayy_train = pd.DataFrame(index=txtIndex, data=ayy_train, columns=colTitles)  
            dfResul_ayy_train.index.name = 'Run'  
            dfResul_ayy_train.to_csv(nameFile, header=True, index=True)  
            return [ayy_train, dfResul_ayy_train]

        def save_County_Statistics(arrayMSE_County_Train, txtIndex, txt_FIPS, nameFile, DGP_num_nodes):
            arrayMSE_County_Train = np.concatenate((arrayMSE_County_Train, np.mean(arrayMSE_County_Train, axis=0).reshape(1, DGP_num_nodes))) # Mean
            arrayMSE_County_Train = np.concatenate((arrayMSE_County_Train, np.std(arrayMSE_County_Train, axis=0).reshape(1, DGP_num_nodes))) # STD
            dfResul_MSE_County_train = pd.DataFrame(index=txtIndex, data=arrayMSE_County_Train, columns=txt_FIPS)
            dfResul_MSE_County_train.index.name = 'Run'
            dfResul_MSE_County_train.to_csv(nameFile, header=True, index=True)
            return [arrayMSE_County_Train, dfResul_MSE_County_train] 

        def save_Time_Run(ayy_train, txtIndex, colTitles, nameFile):
            ayy_train = np.concatenate((ayy_train, np.mean(ayy_train, axis=0).reshape(1,1)))  # Mean 
            ayy_train = np.concatenate((ayy_train, np.std(ayy_train, axis=0).reshape(1,1)))  # Std 
            dfResul_ayy_train = pd.DataFrame(index=txtIndex, data=ayy_train, columns=colTitles)  
            dfResul_ayy_train.index.name = 'Run'  
            dfResul_ayy_train.to_csv(nameFile, header=True, index=True)  
            return [ayy_train, dfResul_ayy_train]



        #%% Open Datasets
        # Open Adjacency Matrix and County
        df_ADJ = pd.read_csv(pathInputALL+nameFileADJ) # Adjacency matrix
        df_County = pd.read_csv(pathInputALL+nameFileCounties) # List of Counties
        # The target Serie
        df_Target_Aux = pd.read_csv(pathInputALL+sel_TARGET, index_col=0) # Selected target
        #if((sel_TARGET==nameIN_CASES) or (sel_TARGET==nameIN_DEATH) or (sel_TARGET==nameIN_HOSPI)):  
        if((sel_TARGET==nameIN_CASES) or (sel_TARGET==nameIN_DEATH)):  
            df_Target = getDaily_Dataframe(df_Target_Aux, initialDay, finalDay)
        elif(sel_TARGET==nameIN_AOD):
            df_Target = replace_greater_Dataframe(df_Target_Aux, 5)
        else:
            df_Target = df_Target_Aux.copy() 
        # To choose only the selected period of time
        df_Target = getSelected_Period_Dataframe(df_Target.copy(), initialDay, finalDay)

        # Open Features
        lis_DF_FEATURES = []
        for k in range(len(sel_FEATURES)):
            dfData = pd.read_csv(pathInputALL+sel_FEATURES[k], index_col=0)
            if((sel_FEATURES[k]==nameIN_CASES) or (sel_FEATURES[k]==nameIN_DEATH)):
                dfData_NewD = getDaily_Dataframe(dfData, initialDay, finalDay)
            elif(sel_FEATURES[k]==nameIN_AOD):
                dfData_NewD = replace_greater_Dataframe(dfData, 5)
            else:
                dfData_NewD = dfData.copy()
            # To choose only the selected period of time
            dfData_NewD = getSelected_Period_Dataframe(dfData_NewD.copy(), initialDay, finalDay)
            # Append New Dataframe/Dataset  
            lis_DF_FEATURES.append(dfData_NewD.copy())

        #%% STANDARIZATION of datasets
        ## IMPORTANT !!!
        ## IT SHOULD BE DONE ONLY USING TRAIN DATA OF DATASET  
        # --- Range of dates
        datesRange = pd.date_range(start=initialDay, end=finalDay, freq='D') 
        datesRangeTXT = list(datesRange.strftime('%m/%d/%Y'))
        # --- To compute upper-index for training 
        HIFO_NGraphs_aux = len(datesRange) - LForward - LBack
        indTop_Train = int(percenTrain*HIFO_NGraphs_aux) # For Converted Graphs
        ind_Before_X = indTop_Train + LBack 
        ind_Before_Y = indTop_Train + LBack + LForward 

        # --- Standarization
        # Target
        df_Target_Orig = df_Target.copy()  # Copy of Original Data
        [scaler_Target, df_Target] = get_Standarized_Dataframe(df_Target.copy(), ind_Before_Y)  

        # Features
        lis_DF_FEATURES_Orig = []
        lis_Feat_SCALER = []
        for k in range(len(lis_DF_FEATURES)):
            lis_DF_FEATURES_Orig.append( lis_DF_FEATURES[k].copy() ) # Copy of Original Data
            [scaler_Aux, df_Out_Aux] = get_Standarized_Dataframe(lis_DF_FEATURES[k].copy(), ind_Before_X)      
            # eplacing and Saving data
            lis_DF_FEATURES[k] = df_Out_Aux.copy()
            lis_Feat_SCALER.append( scaler_Aux )

        #%% Build Data for each Day
        DATABASE_lis = []
        # Dynamic Graph Parameters
        DGP_num_nodes = df_County.shape[0]
        DGP_num_node_features = len(lis_DF_FEATURES)
        DGP_num_edges = df_ADJ.shape[0]
        DGP_NGraphs = len(datesRange)
        # Graph connectivity in COO format with shape [2, num_edges] and type torch.long 
        array_ADJ = np.concatenate((df_ADJ[['From','To']].values, df_ADJ[['To','From']].values))
        tensor_edge_index = torch.from_numpy(array_ADJ).t().contiguous()
        # Edge feature matrix 
        tensor_edge_attr = torch.tensor([1.0]*tensor_edge_index.shape[1], dtype=torch.float)
        # Build DAILY data: feature matrix 'x' and target 'y'  
        for txtDay in tqdm(datesRangeTXT):  
            # Target 'y' using value  
            lisY = [1]*df_County.shape[0]  # List same size of number of counties
            for iC in range(df_County.shape[0]):  
                val_Y = float(df_Target[str(df_County.iloc[iC].FIPS)].loc[txtDay])
                lisY[iC] = val_Y
            tensor_Y = torch.tensor(lisY, dtype=torch.float) 
            # Target 'y' using REAL value  
            lisY_real = [1]*df_County.shape[0]  # List same size of number of counties
            for iC in range(df_County.shape[0]):  
                val_Y = float(df_Target_Orig[str(df_County.iloc[iC].FIPS)].loc[txtDay])
                lisY_real[iC] = val_Y
            tensor_Y_real = torch.tensor(lisY_real, dtype=torch.float) 
            # Node feature matrix 'x' 
            lisX = []
            for iC in range(df_County.shape[0]):  
                aux_X = [0]*len(sel_FEATURES)
                for kF in range(len(sel_FEATURES)): 
                    valAux = float(lis_DF_FEATURES[kF][str(df_County.iloc[iC].FIPS)].loc[txtDay]) 
                    aux_X[kF] = valAux
                lisX.append(aux_X)
            tensor_X = torch.tensor(lisX, dtype=torch.float)
            # Save Graph in List
            DATABASE_lis.append( Data(edge_index=tensor_edge_index, edge_attr=tensor_edge_attr, x=tensor_X,  y=tensor_Y, y_real=tensor_Y_real) )  

        #%%
        #***************************************
        # Here I should modify 'DATABASE_lis'
        # to add lagged information
        # New variable: 'LBack'
        # Modify variable: 'DGP_num_node_features' and 'DGP_NGraphs'
        #***************************************
        # Modify variable 'Y' 
        # New variable: 'LForward'  (1 month ahead, 15 days ahead, etc)
        #***************************************
        # Create a new DATABASE_lis

        #%% Build new DATABASE including HISTORICAL and FORECASTING
        DATABASE_HISFORE = []
        for kB in range(LBack, DGP_NGraphs-LForward):
            # Forecasting Y  
            tensor_YFore = DATABASE_lis[kB+LForward].y 
            # Forecasting Y REAL 
            tensor_YFore_real = DATABASE_lis[kB+LForward].y_real 
            # Rebuild X features  
            torch_XCat = torch.empty(DGP_num_nodes, 0)
            for iH in range(kB-LBack,kB+1):
                torch_XCat = torch.cat((torch_XCat, DATABASE_lis[iH].x), 1)
            # To add Merged Features and Forecasting Y
            DATABASE_HISFORE.append( Data(edge_index=tensor_edge_index, edge_attr=tensor_edge_attr, x=torch_XCat,  y=tensor_YFore, y_real=tensor_YFore_real) )
        # Some general variables will change
        HIFO_num_node_features = len(lis_DF_FEATURES)*(LBack+1) 
        HIFO_NGraphs = len(DATABASE_HISFORE)  
        HIFO_num_nodes = df_County.shape[0]
        HIFO_num_edges = df_ADJ.shape[0]

        #%% Separate in TRAIN and TEST  
        indTop_Train = int(percenTrain*HIFO_NGraphs)
        train_dataset = DATABASE_HISFORE[0:indTop_Train]
        test_dataset = DATABASE_HISFORE[indTop_Train:HIFO_NGraphs]

        #%% To define a Recurrent Graph Neural Network architecture
        ## One layer....
        class RecurrentGCN(torch.nn.Module):  
            def __init__(self, node_features, node_count):  
                super(RecurrentGCN, self).__init__()  
                self.recurrent = MPNNLSTM(node_features, 128, 128, node_count, 1, 0.5)  
                self.linear = torch.nn.Linear(2*128 + node_features, 1)  

            def forward(self, x, edge_index, edge_weight):
                h = self.recurrent(x, edge_index, edge_weight)
                h = F.dropout(h, p=0.5, training=self.training)
                h = F.relu(h)
                h = self.linear(h)
                return h


        #%% To train and test several times
        # RMSE
        arrayMSE = np.zeros((NTT, 2)) # To save MSE
        arrayMSE_County_Train = np.zeros((NTT, HIFO_num_nodes)) # To save MSE 
        arrayMSE_County_test = np.zeros((NTT, HIFO_num_nodes)) # To save MSE 
        # Over-Under Predction
        ayy_train = np.zeros((NTT, 2)) # To save Total OVER-UNDER Prediction (TRAIN)
        ayy_test = np.zeros((NTT, 2)) # To save Total OVER-UNDER Prediction (TEST)
        ayyOVER_Train = np.zeros((NTT, HIFO_num_nodes)) # To save COUNTY OVER Prediction (TRAIN)
        ayyUNDER_Train = np.zeros((NTT, HIFO_num_nodes)) # To save COUNTY UNDER Prediction (TRAIN)
        ayyOVER_Test = np.zeros((NTT, HIFO_num_nodes)) # To save COUNTY OVER Prediction (TEST)
        ayyUNDER_Test = np.zeros((NTT, HIFO_num_nodes)) # To save COUNTY UNDER Prediction (TEST) 
        # Time
        vecTime = np.zeros((NTT, 1)) # To save Time 
        # --- For everyything 
        for iTT in range(NTT):
            print(f'  ----------- Run [{iTT}] ----------- ')
            #*** Time ***
            run_time = TIME.time() # To measure time (RUN)   

            #%% --- Training 
            model = RecurrentGCN(node_features = HIFO_num_node_features, node_count=HIFO_num_nodes)  
            #optimizer = torch.optim.Adam(model.parameters(), lr=0.025)
            #optimizer = torch.optim.AdamW(model.parameters(), lr=0.05, weight_decay=0.001, amsgrad=False)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.02, amsgrad=True)
            model.train()  
            #%% Training
            for epoch in tqdm(range( NEpochs )):   # make your loops show a smart progress meter 
                cost = 0
                #H_state = None
                for timeCount, snapshot in enumerate(train_dataset):
                    y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
                    cost = cost + torch.sqrt( torch.mean((y_hat[0].t().contiguous()-snapshot.y)**2) )
                cost = cost / (timeCount+1)
                #print("Training >>> Final MSE: {:.10f}".format(cost.item()))
                cost.backward()
                optimizer.step()
                optimizer.zero_grad()  
            #print("Training >>> Final RMSE (standarized): {:.4f}".format(cost.item()))  
            
            #%% Evaluation of model on TRAINING Set
            model.eval() 
            matPred_Train_hat = np.zeros((len(train_dataset), HIFO_num_nodes)) ### To Save the ESTIMATION of values
            matReal_Train_Y = np.zeros((len(train_dataset), HIFO_num_nodes)) ### To Save the ESTIMATION of values
            for timeCount, snapshot in enumerate(train_dataset):
                y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
                matPred_Train_hat[timeCount] = y_hat.t().contiguous().detach().numpy()[0]
                matReal_Train_Y[timeCount] = snapshot.y_real.detach().numpy()   ### Y

            #%% Evaluation of model on TESTING Set
            model.eval() 
            matPred_Test_hat = np.zeros((len(test_dataset), HIFO_num_nodes)) ### To Save the ESTIMATION of values
            matReal_Test_Y = np.zeros((len(test_dataset), HIFO_num_nodes)) ### To Save the ESTIMATION of values
            for timeCount, snapshot in enumerate(test_dataset):
                y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
                matPred_Test_hat[timeCount] = y_hat.t().contiguous().detach().numpy()[0]
                matReal_Test_Y[timeCount] = snapshot.y_real.detach().numpy()   ### Y
            
            #*** Time ***
            vecTime[iTT] = TIME.time() - run_time

            #%% To invert standarization
            train_Y = matReal_Train_Y   
            train_hat = scaler_Target.inverse_transform(matPred_Train_hat) 
            test_Y = matReal_Test_Y  
            test_hat = scaler_Target.inverse_transform(matPred_Test_hat)

            #%% To compute RMSE
            arrayMSE[iTT, 0] = np.sum( np.sqrt( np.mean((train_Y-train_hat)**2, axis=1) ) )/(train_hat.shape[0] + 1)
            arrayMSE[iTT, 1] = np.sum( np.sqrt( np.mean((test_Y-test_hat)**2, axis=1) ) )/(test_hat.shape[0] + 1)
            arrayMSE_County_test[iTT] = np.sqrt(np.sum((test_Y-test_hat)**2, axis=0)/(test_hat.shape[0] + 1) )
            arrayMSE_County_Train[iTT] = np.sqrt(np.sum((train_Y-train_hat)**2, axis=0)/(train_hat.shape[0] + 1) )  
            
            #%% To compute OVER & UNDER Prediction on TRAIN
            # At County Level
            subsHatY_train = train_hat - train_Y
            ayyOVER_Train[iTT] = 100*np.sum(subsHatY_train>0, axis=0)/subsHatY_train.shape[0]
            ayyUNDER_Train[iTT] = 100*np.sum(subsHatY_train<0, axis=0)/subsHatY_train.shape[0]
            subsHatY_test = test_hat - test_Y  
            ayyOVER_Test[iTT] = 100*np.sum(subsHatY_test>0, axis=0)/subsHatY_test.shape[0]
            ayyUNDER_Test[iTT] = 100*np.sum(subsHatY_test<0, axis=0)/subsHatY_test.shape[0]
            # Total
            ayy_train[iTT, 0] = 100*np.sum(subsHatY_train>0)/subsHatY_train.size  # OVER Prediction
            ayy_train[iTT, 1] = 100*np.sum(subsHatY_train<0)/subsHatY_train.size  # UNDER Prediction
            ayy_test[iTT, 0] = 100*np.sum(subsHatY_test>0)/subsHatY_test.size  # OVER Prediction
            ayy_test[iTT, 1] = 100*np.sum(subsHatY_test<0)/subsHatY_test.size  # UNDER Prediction

            # To save Prediction and Forecasting
            pd.DataFrame(data=train_hat, columns=[str(a) for a in df_County.FIPS]).to_csv(pathOutput+'IMGS/matTRAIN_hat_'+str(iTT)+'.csv', index=False)   
            pd.DataFrame(data=train_Y, columns=[str(a) for a in df_County.FIPS]).to_csv(pathOutput+'IMGS/matTRAIN_Y_'+str(iTT)+'.csv', index=False)   
            pd.DataFrame(data=test_hat, columns=[str(a) for a in df_County.FIPS]).to_csv(pathOutput+'IMGS/matTEST_hat_'+str(iTT)+'.csv', index=False)   
            pd.DataFrame(data=test_Y, columns=[str(a) for a in df_County.FIPS]).to_csv(pathOutput+'IMGS/matTEST_Y_'+str(iTT)+'.csv', index=False)   

            # To Print Outout
            print("Training >>> Final RMSE : {:.4f}".format(arrayMSE[iTT, 0]))  
            print("TEST >>> Final RMSE : {:.4f}".format(arrayMSE[iTT, 1]))  


        # Convert to dataframe and save in CSV 
        txtIndex = [str(a) for a in range(NTT)]
        txtIndex.append('average')
        txtIndex.append('std')  
        # Save to CSV 
        nameFile_CSV = ''
        for kN in range(len(sel_FEATURES)):
            nameFile_CSV = nameFile_CSV + sel_FEATURES[kN].split('_')[1]+'_'
        nameFile_CSV = nameFile_CSV + sel_TARGET.split('_')[1].lower() + '.csv'
        txt_FIPS = [str(a) for a in df_County.FIPS] # FIPS

        # --- RMSE
        # Total
        [arrayMSE, dfResul_MSE] = save_Total_Statistics(arrayMSE, txtIndex, ['Train_RMSE', 'Test_RMSE'], pathOutput+'IMGS/Total_RMSE_'+nameFile_CSV)
        # County ( Training )
        [arrayMSE_County_Train, dfResul_MSE_County_train] = save_County_Statistics(arrayMSE_County_Train, txtIndex, txt_FIPS, pathOutput+'IMGS/RMSE_county_TRAIN_'+nameFile_CSV, DGP_num_nodes)
        # County ( Test ) 
        [arrayMSE_County_test, dfResul_MSE_County_test] = save_County_Statistics(arrayMSE_County_test, txtIndex, txt_FIPS, pathOutput+'IMGS/RMSE_county_TEST_'+nameFile_CSV, DGP_num_nodes)

        # --- OVER-UNDER Prediction 
        # Total (Train)
        [ayy_train, dfResul_ayy_train] = save_Total_Statistics(ayy_train, txtIndex, ['Percen_OVER', 'Percen_UNDER'], pathOutput+'IMGS/Total_OVER_UNDER_Pred_TRAIN_'+nameFile_CSV)
        # Total (Test)
        [ayy_test, dfResul_ayy_test] = save_Total_Statistics(ayy_test, txtIndex, ['Percen_OVER', 'Percen_UNDER'], pathOutput+'IMGS/Total_OVER_UNDER_Pred_TEST_'+nameFile_CSV)
        # County ( Train )  
        [ayyOVER_Train, dfResul_ayyOVER_Train] = save_County_Statistics(ayyOVER_Train, txtIndex, txt_FIPS, pathOutput+'IMGS/OVER_county_TRAIN_'+nameFile_CSV, DGP_num_nodes)
        [ayyUNDER_Train, dfResul_ayyUNDER_Train] = save_County_Statistics(ayyUNDER_Train, txtIndex, txt_FIPS, pathOutput+'IMGS/UNDER_county_TRAIN_'+nameFile_CSV, DGP_num_nodes)
        # County ( Test )  
        [ayyOVER_Test, dfResul_ayyOVER_Test] = save_County_Statistics(ayyOVER_Test, txtIndex, txt_FIPS, pathOutput+'IMGS/OVER_county_TEST_'+nameFile_CSV, DGP_num_nodes)
        [ayyUNDER_Test, dfResul_ayyUNDER_Test] = save_County_Statistics(ayyUNDER_Test, txtIndex, txt_FIPS, pathOutput+'IMGS/UNDER_county_TEST_'+nameFile_CSV, DGP_num_nodes)

        # --- Time of Training-Runs 
        [vecTime, df_vecTime] = save_Time_Run(vecTime, txtIndex, ['Time_Seconds'], pathOutput+'IMGS/TIME_SECONDS_'+nameFile_CSV)


        #%% Final Time
        print("RUNNING TIME: "+str((TIME.time() - start_time))+" Seg ---  "+str((TIME.time() - start_time)/60)+" Min ---  "+str((TIME.time() - start_time)/(60*60))+" Hr ") 
        print('DONE !!!')

        #%%
