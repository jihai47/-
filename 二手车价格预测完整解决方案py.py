import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
import joblib
import warnings
import os
import numpy as np # 添加这一行
import lightgbm as lgb # 添加 LightGBM 导入

warnings.filterwarnings('ignore')

class CarPricePredictor:
    def __init__(self, model_path='car_model.joblib'):
        self.model_path = model_path
        self.model = None
        self.features = ['car_age', 'km', 'brand', 'body', 'fuel', 'gear', 'power', 'damage'] + [f'v_{i}' for i in range(15)]
        
    def preprocess(self, df):
        """预处理数据，确保不修改原始数据"""
        df = df.copy()
        
        # 将日期列转换为datetime对象，处理错误
        df['creatDate_dt'] = pd.to_datetime(df['creatDate'], errors='coerce')
        df['regDate_dt'] = pd.to_datetime(df['regDate'], errors='coerce')

        # 1. 时间特征处理
        if 'creatDate_dt' in df.columns and 'regDate_dt' in df.columns:
            # 计算车龄（天）
            df['car_age'] = (df['creatDate_dt'] - df['regDate_dt']).dt.days.fillna(0)
            # 提取创建日期的年份、月份、星期几
            df['creatDate_year'] = df['creatDate_dt'].dt.year.fillna(0)
            df['creatDate_month'] = df['creatDate_dt'].dt.month.fillna(0)
            df['creatDate_weekday'] = df['creatDate_dt'].dt.weekday.fillna(0)
            # 提取注册日期的年份、月份、星期几
            df['regDate_year'] = df['regDate_dt'].dt.year.fillna(0)
            df['regDate_month'] = df['regDate_dt'].dt.month.fillna(0)
            df['regDate_weekday'] = df['regDate_dt'].dt.weekday.fillna(0)
            # 计算注册到创建的月份差
            df['reg_creat_month_diff'] = ((df['creatDate_dt'].dt.year - df['regDate_dt'].dt.year) * 12 + \
                                          (df['creatDate_dt'].dt.month - df['regDate_dt'].dt.month)).fillna(0)
        else:
            print("警告：'creatDate' 或 'regDate' 列缺失，无法计算时间特征。相关特征将设置为 0。")
            df['car_age'] = 0
            df['creatDate_year'] = 0
            df['creatDate_month'] = 0
            df['creatDate_weekday'] = 0
            df['regDate_year'] = 0
            df['regDate_month'] = 0
            df['regDate_weekday'] = 0
            df['reg_creat_month_diff'] = 0

        if 'kilometer' in df.columns:
            df['km'] = (
                df['kilometer']
                .astype(str)
                .str.replace('万', 'e4', regex=False)
                .str.replace('公里', '', regex=False)
                .str.replace(',', '', regex=False)
                .apply(pd.to_numeric, errors='coerce') 
            )
        else:
            print("警告：'kilometer' 列缺失，'km' 将被设置为 0。")
            df['km'] = 0.0
        
        rename_map = {
            'bodyType': 'body',
            'fuelType': 'fuel',
            'gearbox': 'gear',
            'notRepairedDamage': 'damage'
        }
        df.rename(columns=rename_map, inplace=True)
        
        # 确保所有必要的列都存在，如果不存在则添加并填充默认值
        # 这一步放在特征生成之前，确保后续操作不会因为列缺失而报错
        required_cols_for_features = [
            'car_age', 'km', 'power', 'brand', 'body', 'fuel', 'gear', 'damage',
            'creatDate_year', 'creatDate_month', 'creatDate_weekday',
            'regDate_year', 'regDate_month', 'regDate_weekday', 'reg_creat_month_diff'
        ] + [f'v_{i}' for i in range(15)]

        for col in required_cols_for_features:
            if col not in df.columns:
                df[col] = 0.0 # 默认填充为0，后续可能需要更智能的填充
                print(f"警告：特征列 '{col}' 缺失，已添加并填充为 0.0。")

        # 确保数值列是数值类型并处理NaN
        num_features_to_process = [
            'car_age', 'km', 'power',
            'creatDate_year', 'creatDate_month', 'creatDate_weekday',
            'regDate_year', 'regDate_month', 'regDate_weekday', 'reg_creat_month_diff'
        ] + [f'v_{i}' for i in range(15)]
        for col in num_features_to_process:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        
        # 2. 数值特征的对数变换
        for col in ['km', 'power']:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: np.log1p(x) if x >= 0 else x) 

        # 3. 交互特征
        df['power_div_km'] = df['power'] / (df['km'] + 1e-6) # 避免除以0
        df['age_x_power'] = df['car_age'] * df['power']
        df['age_x_km'] = df['car_age'] * df['km']
        
        # 4. 匿名特征的统计量
        v_features = [f'v_{i}' for i in range(15)]
        df['v_sum'] = df[v_features].sum(axis=1)
        df['v_mean'] = df[v_features].mean(axis=1)
        df['v_std'] = df[v_features].std(axis=1).fillna(0) # 填充std的NaN，例如只有一列时std为NaN
        df['v_median'] = df[v_features].median(axis=1)
        df['v_min'] = df[v_features].min(axis=1)
        df['v_max'] = df[v_features].max(axis=1)

        # 更新特征列表以包含所有新生成的特征
        self.features = [
            'car_age', 'km', 'brand', 'body', 'fuel', 'gear', 'power', 'damage',
            'creatDate_year', 'creatDate_month', 'creatDate_weekday',
            'regDate_year', 'regDate_month', 'regDate_weekday', 'reg_creat_month_diff',
            'power_div_km', 'age_x_power', 'age_x_km',
            'v_sum', 'v_mean', 'v_std', 'v_median', 'v_min', 'v_max'
        ] + [f'v_{i}' for i in range(15)] # 保持原始v_特征

        return df[self.features]
    
    def build_model(self):
        """构建预处理和模型管道"""
        # 更新数值特征列表以包含新的时间特征、交互特征和v_特征的统计量
        num_features = [
            'car_age', 'km', 'power',
            'creatDate_year', 'creatDate_month', 'creatDate_weekday',
            'regDate_year', 'regDate_month', 'regDate_weekday', 'reg_creat_month_diff',
            'power_div_km', 'age_x_power', 'age_x_km',
            'v_sum', 'v_mean', 'v_std', 'v_median', 'v_min', 'v_max'
        ] + [f'v_{i}' for i in range(15)]
        
        cat_features = ['brand', 'body', 'fuel', 'gear', 'damage']
        
        preprocessor = ColumnTransformer([
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')), 
                ('scaler', StandardScaler())
            ]), num_features),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('encoder', OneHotEncoder(handle_unknown='ignore', sparse=False))
            ]), cat_features)
        ])
        
        return Pipeline([
            ('preprocessor', preprocessor),
            ('model', lgb.LGBMRegressor(
                objective='regression_l1', # 使用MAE作为优化目标，通常对异常值更鲁棒
                metric='mae', # 评估指标
                n_estimators=1000000, # 进一步大幅增加迭代次数
                learning_rate=0.001, # 进一步大幅增加迭代次数
                num_leaves=63, # 增加叶子节点数量
                max_depth=-1, # 无限制深度
                min_child_samples=10, # 减小最小样本数
                subsample=0.8, # 训练每棵树时随机采样的样本比例
                colsample_bytree=0.8, # 训练每棵树时随机采样的特征比例
                random_state=42,
                n_jobs=-1, # 使用所有可用核心
                reg_alpha=0.1, # L1正则化
                reg_lambda=0.1, # L2正则化
                # early_stopping_rounds=100 # 确保这一行是被注释掉的
            ))
        ])
    
    def train(self, train_path, test_path=None):
        """训练并评估模型"""
        train_df = pd.read_csv(train_path, sep='\s+')
        print(f"训练数据文件 '{train_path}' 加载后的列名: {train_df.columns.tolist()}")
        
        y_train = train_df['price']
        X_train = self.preprocess(train_df)
        
        self.model = self.build_model()
        self.model.fit(X_train, y_train)
        
        if test_path:
            test_df = pd.read_csv(test_path, sep='\s+') # 确保测试集也使用正确的sep
            if 'price' in test_df.columns:
                X_test = self.preprocess(test_df)
                y_test = test_df['price']
                y_pred = self.model.predict(X_test)
                mae = mean_absolute_error(y_test, y_pred)
                print(f"模型训练完成，测试MAE: {mae:,.2f} 元")
            else:
                print("警告：测试集文件不包含 'price' 列，跳过 MAE 计算。")
        
        joblib.dump(self.model, self.model_path)
        print(f"模型已保存至 {self.model_path}")
    
    def load_model(self):
        """加载预训练模型"""
        self.model = joblib.load(self.model_path)
        print("模型加载成功")
        return self
        
    def predict(self, input_data):
        """预测二手车价格"""
        if self.model is None:
            self.load_model()
            
        if isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data])
        else:
            input_df = input_data.copy()
            
        preprocessed = self.preprocess(input_df)
        return self.model.predict(preprocessed)

if __name__ == "__main__":
    predictor = CarPricePredictor()
    
    if not os.path.exists(predictor.model_path):
        print("模型文件不存在，开始训练模型...")
        predictor.train('used_car_train_20200313.csv', 'used_car_testA_20200313.csv')
    else:
        print("模型文件已存在，加载模型...")
        predictor.load_model()
    
    sample_car = {
        'regDate': '2018-05-01',
        'creatDate': '2023-10-15',
        'kilometer': '5.5万公里',
        'brand': 'BMW',
        'bodyType': 'SUV',
        'fuelType': 'Gasoline',
        'gearbox': 'Auto',
        'power': 184,
        'notRepairedDamage': 0,
        **{f'v_{i}': 0.5 for i in range(15)}
    }
    
    price = predictor.predict(sample_car)[0]
    print(f"预测价格: {price:,.2f} 元")

    print("\n开始生成提交文件...")
    try:
        test_A_df = pd.read_csv('used_car_testA_20200313.csv', sep='\s+')
        test_A_ids = test_A_df['SaleID']
        
        test_A_predictions = predictor.predict(test_A_df)
        
        submission_df = pd.DataFrame({
            'SaleID': test_A_ids,
            'price': test_A_predictions
        })
        
        submission_file_path = 'submission.csv'
        submission_df.to_csv(submission_file_path, index=False)
        print(f"提交文件已成功生成至 {submission_file_path}")
        print("提交文件前5行：")
        print(submission_df.head())
        
    except Exception as e:
        print(f"生成提交文件时发生错误: {e}")