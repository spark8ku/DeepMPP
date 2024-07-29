from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, RobustScaler

class identityScaler:
    def fit(self,X):
        pass
    def transform(self,X):
        return X
    def fit_transform(self,X):
        return X
    def inverse_transform(self,X):
        return X


class Scaler:
    def __init__(self,scale_type="standard"):
        self.scale_type = scale_type
        if self.scale_type=='standard':
            self.scaler = StandardScaler()
        elif self.scale_type=='minmax':
            self.scaler = MinMaxScaler()
        elif self.scale_type=='normalizer':
            self.scaler = Normalizer()
        elif self.scale_type=='robust':
            self.scaler = RobustScaler()
        else:
            self.scaler = identityScaler()
            
    def fit(self,X):
        self.scaler.fit(X)

    def transform(self,X):
        return self.scaler.transform(X)

    def fit_transform(self,X):
        return self.scaler.fit_transform(X)
    
    def inverse_transform(self,X):
        return self.scaler.inverse_transform(X)
    

