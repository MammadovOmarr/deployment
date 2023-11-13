import pandas as pd
import numpy as np
import plotly.express as px
from PIL import Image
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import RandomOverSampler
import streamlit as st
from sklearn.preprocessing import StandardScaler , MinMaxScaler , RobustScaler
from sklearn.preprocessing import LabelEncoder
import os
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score , confusion_matrix , classification_report , roc_curve , roc_auc_score

icon = Image.open("logo.png")
logo = Image.open("51c67a52794145.591cb02edf439.jpg")
image = Image.open("homepage.jpg")

st.set_page_config(layout='wide',
                   page_title='deployment',
                   page_icon=icon)
st.title('**Deployment** - ML models')
st.text('Deployment of loan and water protabibilty datas')

st.sidebar.image(image=logo)

menu = st.sidebar.selectbox('', ['Homepage' , 'EDA' , 'Modelling'])

if menu == "Homepage":
    st.header('Homepage')
    st.image(image, use_column_width='always')

    dataset = st.selectbox('Select dataset', ["Loan prediction", "Water probability"])
    st.markdown(f'Selected **{dataset}** Dataset')

    if dataset == 'Loan prediction':
        st.warning('Selected **Loan prediction** dataset')
        st.info("""**Loan_ID** - Unique Loan id \n
                **Gender** - Male/Female \n
                **Married** - Applicant married (Y/N) \n
                **Dependents** - Number of dependents \n
                **Education** - Applicant Education (Graduate/Under Graduate) \n
                *Self_Employed** - Self-employed (Y/N) \n
                **ApplicantIncome** - Applicant income \n
                **CoapplicantIncome** - Coapplicant income \n
                **LoanAmount** - Loan amount in thousands \n
                **LoanAmount_Term** - Term of loan in months \n
            **CreditHistory** - Credit history meets guidelines \n
            **LoanStatus** - Target loan approved (Y/N) \n""")

    else:
        st.warning('Selected **Water probability** dataset')
        st.info("""**pH** - The pH Level of the water. \n
    **Hardness** - Water hardness, a measure of mineral content. \n
    **Solids** - Total dissolved solids in the water. \n
    **Chloramines** - Chloramines concentration in the water. \n
    **Sulfate** - Sulfate concentration in the water. \n
    **Conductivity** - Electrical conductivity of the water. \n
    **OrganicCarbon** - Organic carbon content in the water. \n
    **Trihalomethanes** - Trihalomethanes concentration in the water. \n
    **Turbidity** - Turbidity level, a measure of water clarity. \n
    **Potability** - Target variable; indicates water potability with values 1 (potable) and 0 (not potable). \n""")

elif menu == 'EDA':
    def outliers_thresholds(col_name, q1=0.25, q3=0.75):
        col_name = sorted(col_name)  # Fixed the sorting

        inter_quartile_range = q3 - q1  # Fixed the subtraction order
        lower_bound = q1 - 1.5 * inter_quartile_range
        upper_bound = q3 + 1.5 * inter_quartile_range
        return lower_bound, upper_bound

    def describe_data(df):
        st.dataframe(df)

        st.subheader('Statistical Values')
        st.dataframe(df.describe().T)

        st.subheader('Balance of Data')
        st.bar_chart(df.iloc[:, -1].value_counts())

        null_df = df.isnull().sum().to_frame().reset_index()
        null_df.columns = ['Columns', 'Counts']

        p1, p2, p3 = st.columns([2, 1, 2])

        p1.subheader('Null variables')
        p1.dataframe(null_df)

        p2.subheader('Imputation')
        cat_m = p2.radio('Categorical', ['Mode', 'Backfill', 'Ffill'])
        num_m = p2.radio('Numerical', ['Mode', 'Median'])

        p2.subheader('Feature engineering')
        balance_problem = p2.checkbox('Over Sampling')
        outlier_problem = p2.checkbox('Clean Outliers')

        if p2.button("Data preprocessing"):
            cat_cols = df.select_dtypes(include="object").columns
            num_cols = df.select_dtypes(exclude="object").columns
            if cat_cols.size > 0:
                if cat_m == "Mode":
                    imp_cat = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
                    df[cat_cols] = imp_cat.fit_transform(df[cat_cols])
                elif cat_m == "Backfill":
                    df[cat_cols] = df[cat_cols].fillna(method="backfill")
                else:
                    df[cat_cols] = df[cat_cols].fillna(method="ffill")

            if num_cols.size > 0:
                if num_m == 'Mode':
                    df[num_cols] = df[num_cols].fillna(method="backfill")
                else:
                    df[num_cols] = df[num_cols].fillna(method="ffill")

            if balance_problem:
                oversample = RandomOverSampler()
                x = df.iloc[:, :-1]
                y = df.iloc[:, -1]
                x_resampled, y_resampled = oversample.fit_resample(x, y)
                df = pd.concat([x_resampled, y_resampled], axis=1)

            if outlier_problem:
                for col in num_cols:
                    lower_bound, upper_bound = outliers_thresholds(df[col], q1=0.25, q3=0.75)
                    df[col] = np.clip(df[col], a_min=lower_bound, a_max=upper_bound)

        null_df = df.isnull().sum().to_frame().reset_index()
        null_df.columns = ['Columns', 'Counts']

        p3.subheader('Null Variables')
        p3.dataframe(null_df)
        st.subheader('Balance of data')
        st.bar_chart(df.iloc[:, -1].value_counts())

        heatmap = px.imshow(df.select_dtypes(exclude='object').corr())
        st.plotly_chart(heatmap)
        st.dataframe(df)

        if os.path.exists("model.csv"):
            os.remove("model.csv")
        df.to_csv("model.csv")

    st.header('Exploratory Data Analysis')
    dataset = st.selectbox('Select dataset', ['Loan prediction', 'Water probability'])

    if dataset == 'Loan prediction':
        df = pd.read_csv('loan_pred.csv')
        describe_data(df)
    else:
        df = pd.read_csv('water_potability.csv')
        describe_data(df)


else:
    if not os.path.exists("model.csv"):
        st.header("Please run preprosessing")

    else:
        df = pd.read_csv("model.csv")
        st.dataframe(df)

        p1 , p2 = st.columns(2)

        p1.subheader('Scaling')
        scaling_method = p1.radio("" , ['MinMaxScaler' , 'StandartScaler' , 'RobusScaler'])

        p2.subheader('Encoder')
        encoder_method = p2.radio("" ,['One-hot' , 'Lable'])

        random_state = p1.text_input("Random State")
        test_size = p2.text_input("Test Size")

        model_select = st.radio("Select model" , ["XGBoost" , "CatBoost"  , "RandomForest"])
        st.markdown(f"You selected **{model_select}** model")

        if st.button("Run model"):
            cat_columns = df.drop(df.iloc[:,-1]).select_dtypes(include='object').columns

            num_columns = df.drop(df.iloc[:,-1]).select_dtypes(exclude= 'object').columns
            y = df.iloc[:, -1]

            if num_columns.size > 0:
                if scaling_method == 'MinMaxScaler':
                    sc = MinMaxScaler()
                elif scaling_method == "StandartScaler":
                    sc = StandardScaler()
                else:
                    sc = RobustScaler()

                df[num_columns] = sc.fit_transform(df[num_columns])

            if cat_columns.size > 0:
                if encoder_method == "Label":
                    lb = LabelEncoder()
                    for col in cat_columns:
                        df[col] = lb.fit_transform(df[col])

                else:
                    df.drop(df.columns[-1], axis=1, inplace=True)

                    get_dummy = pd.get_dummies(df[cat_columns])
                    df.drop(df[get_dummy] , axis=1 , inplace=True)
                    df = pd.concat([df, get_dummy, y], axis=1)


        
            st.dataframe(df)
            x = df.iloc[: , :-1]

            x_train , x_test, y_train , y_test = train_test_split(x ,y , test_size=float(test_size) , random_state=int(random_state))


            if model_select == "XGBoost":
                model = XGBClassifier().fit(x_train , y_train)
            elif model_select == "CatBoost":
                model = CatBoostClassifier().fit(x_train, y_train)

            else:
                model = RandomForestClassifier(max_depth=2 , random_state=int(random_state)).fit(x_train , y_train)
        
            #model.fit(x_train, y_train)

            prediction = model.predict(x_test)
            y_score = model.predict_proba(x_test)


            st.markdown("Confussion matrix")
            st.write(confusion_matrix(y_test,prediction))


            st.markdown("Classification report")
            report = classification_report(y_test, prediction, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df)
            st.markdown("Accuracy score " + str(accuracy_score(y_test,prediction)))