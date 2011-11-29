#include "settings_database.h"

#include <iostream>
#include <iomanip>

namespace Settings {

    ////////////////////////////////////////////////////////////////////////////

    Database::Database() {

    }

    ////////////////////////////////////////////////////////////////////////////

    Database::~Database() {

    }

    ////////////////////////////////////////////////////////////////////////////

    Database* Database::insert(
        RecordType type,
        string name,
        float minimum,
        float maximum,
        float value,
        string unit,
        bool editable
    ) {
        Record record;
        record.name = name;
        record.minimum = minimum;
        record.maximum = maximum;
        record.value = value;
        record.unit = unit;
        record.editable = editable;

        this->_settingsMap[type] = record;
        return this;
    }

    ////////////////////////////////////////////////////////////////////////////

    Database* Database::insert(
        RecordType type,
        string name,
        float minimum,
        float maximum,
        float value,
        bool editable
    ) {
        return this->insert(type,name, minimum, maximum, value, "", editable);
    }

    ////////////////////////////////////////////////////////////////////////////

    Database* Database::insert(
        RecordType type,
        string name,
        float minimum,
        float maximum,
        float value,
        string unit
    ) {
        return this->insert(type,name, minimum, maximum, value, unit, true);
    }

    ////////////////////////////////////////////////////////////////////////////

    Database* Database::insert(
        RecordType type,
        string name,
        float value,
        string unit
    ) {
        return this->insert(type,name, value, value, value, unit, false);
    }

    ////////////////////////////////////////////////////////////////////////////

    Database* Database::insert(
        RecordType type,
        string name,
        float value,
        bool editable
    ) {
        return this->insert(type,name, value, value, value, "", editable);
    }

    ////////////////////////////////////////////////////////////////////////////

    Database* Database::updateMinimum(RecordType type, float minimum) {
        this->_settingsMap[type].minimum = minimum;
        return this;
    }

    ////////////////////////////////////////////////////////////////////////////

    Database* Database::updateMaximum(RecordType type, float maximum) {
        this->_settingsMap[type].maximum = maximum;
        return this;
    }

    ////////////////////////////////////////////////////////////////////////////

    Database* Database::updateValue(RecordType type, float value) {
        if (this->_settingsMap.find(type) == this->_settingsMap.end()) {
            cout << "Warning: SettingsDatabase can't find setting with record "
                 << "with type " << type << endl;
            return this;
        } else {

            float old = this->_settingsMap[type].value;

            if (old != value) {
                this->_settingsMap[type].value = value;

                for (unsigned int i=0; i<this->_callbackVector.size(); i++) {
                    this->_callbackVector[i]->valueChanged(type);
                }
            }
        }
        return this;
    }

    ////////////////////////////////////////////////////////////////////////////

    float Database::selectValue(RecordType type) {
        if (this->_settingsMap.find(type) == this->_settingsMap.end()) {
            cout << "Warning: SettingsDatabase can't find setting with record "
                 << "with type " << type << endl;
            return 0.0f;
        } else {
            return this->_settingsMap[type].value;
        }
    }

    ////////////////////////////////////////////////////////////////////////////

    void Database::addUpdateCallback(UpdateCallback* callback) {
        this->_callbackVector.push_back(callback);
    }

    ////////////////////////////////////////////////////////////////////////////

    void Database::print() {
        Record record;

        cout << endl
             << setfill('#')
             << setw(30) << ""
             << " Settings database "
             << setw(31) << ""
             << setfill(' ')
             << endl
             << endl;

        cout << left
             << setw(20) << "Name"
             << setw(10) << "Value"
             << setw(10) << "Minimum"
             << setw(10) << "Maximum"
             << setw(10) << "Unit"
             << setw(10) << "Editable"
             << setw(10) << "Type"
             << endl;
        for(
            SettingsMap::const_iterator it = this->_settingsMap.begin();
            it != this->_settingsMap.end();
            ++it
        ) {
            record = (Record) it->second;
            cout << left
                 << setprecision (4)
                 << setw(20) << record.name
                 << setw(10) << record.value
                 << setw(10) << record.minimum
                 << setw(10) << record.maximum
                 << setw(10) << record.unit
                 << setw(10) << ((record.editable) ? "Yes" : "No")
                 << setw(10) << it->first
                 << endl;
        }
        cout << endl;

        cout << setfill('#')
             << setw(80) << ""
             << endl
             << endl;

    }

    ////////////////////////////////////////////////////////////////////////////

}
