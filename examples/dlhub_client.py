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
        return pd.DataFrame(r.json())


def log_progress(sequence, every=None, size=None, name='Items'):
    from ipywidgets import IntProgress, HTML, VBox
    from IPython.display import display

    is_iterator = False
    if size is None:
        try:
            size = len(sequence)
        except TypeError:
            is_iterator = True
    if size is not None:
        if every is None:
            if size <= 200:
                every = 1
            else:
                every = int(size / 200)     # every 0.5%
    else:
        assert every is not None, 'sequence is iterator, set every'

    if is_iterator:
        progress = IntProgress(min=0, max=1, value=1)
        progress.bar_style = 'info'
    else:
        progress = IntProgress(min=0, max=size, value=0)
    label = HTML()
    box = VBox(children=[label, progress])
    display(box)

    index = 0
    try:
        for index, record in enumerate(sequence, 1):
            if index == 1 or index % every == 0:
                if is_iterator:
                    label.value = '{name}: {index} / ?'.format(
                        name=name,
                        index=index
                    )
                else:
                    progress.value = index
                    label.value = u'{name}: {index} / {size}'.format(
                        name=name,
                        index=index,
                        size=size
                    )
            yield record
    except:
        progress.bar_style = 'danger'
        raise
    else:
        progress.bar_style = 'success'
        progress.value = index
        label.value = "{name}: {index}".format(
            name=name,
            index=str(index or '?')
        )
