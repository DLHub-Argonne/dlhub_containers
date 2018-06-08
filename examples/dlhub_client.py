import requests
import pandas as pd
import ipywidgets


class DLHub():
    service = "http://dlhub.org:5000/api/v1"
    
    def __init__(self):
        pass
    
    def get_servables(self):
        r = requests.get("{service}/servables".format(service=self.service))
        return pd.DataFrame(r.json())
    
    def get_id_by_name(self, name):
        r = requests.get("{service}/servables".format(service=self.service))
        df_tmp =  pd.DataFrame(r.json())
        serv = df_tmp[df_tmp.name==name]
        return serv.iloc[0]['uuid']
    
    def infer(self, servable_id, data):
        servable_path = '{service}/servables/{servable_id}/run'.format(service=self.service,
                                                                       servable_id=servable_id)
        payload = {"data":data}

        r = requests.post(servable_path, json=data)
        if r.status_code is not 200:
            raise Exception(r)
        return pd.DataFrame(r.json())

