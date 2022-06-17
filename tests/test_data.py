from bagle import data
import pytest

def test_EventDataDict_init():

    # This should throw an exception
    with pytest.raises(Exception):
        data_dict = data.EventDataDict()

    # Create a valid one.
    data_in = {'target': 'OB110462',
               'raL': '17:51:40.19',
               'decL': '-29:53:26.3'}
        
    data_dict = data.EventDataDict(data_in)

    assert data_dict['target'] == data_in['target']
    assert data_dict['raL'] == data_in['raL']
    assert data_dict['decL'] == data_in['decL']
    assert data_dict['phot_files'] == []
    assert data_dict['phot_data'] == []
    assert data_dict['ast_files'] == []
    assert data_dict['ast_data'] == []
    
    return

