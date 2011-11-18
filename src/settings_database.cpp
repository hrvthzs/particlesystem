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

    void Database::insert(
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
    }

    ////////////////////////////////////////////////////////////////////////////

    void Database::insert(
        RecordType type,
        string name,
        float minimum,
        float maximum,
        float value,
        bool editable
    ) {
        this->insert(type,name, minimum, maximum, value, "", editable);
    }

    ////////////////////////////////////////////////////////////////////////////

    void Database::insert(
        RecordType type,
        string name,
        float minimum,
        float maximum,
        float value,
        string unit
    ) {
        this->insert(type,name, minimum, maximum, value, unit, true);
    }

    ////////////////////////////////////////////////////////////////////////////

    void Database::updateMinimum(RecordType type, float minimum) {
        this->_settingsMap[type].minimum = minimum;
    }

    ////////////////////////////////////////////////////////////////////////////

    void Database::updateMaximum(RecordType type, float maximum) {
        this->_settingsMap[type].maximum = maximum;
    }

    ////////////////////////////////////////////////////////////////////////////

    void Database::updateValue(RecordType type, float value) {
        if (this->_settingsMap.find(type) == this->_settingsMap.end()) {
            cout << "Warning: SettingsDatabase can't find setting with record "
            << "with type " << type << endl;
            return;
        } else {

            float old = this->_settingsMap[type].value;

            if (old != value) {
                this->_settingsMap[type].value = value;

                // TODO callbacks
                /*CallbackVector::const_iterator it;
                for(
                    it = this->_callbackVector.begin();
                    it != this->_callbackVector.end();
                    ++it
                ) {
                    (*it)->valueChanged(type);
                }*/

                for (unsigned int i=0; i<this->_callbackVector.size(); i++) {
                    this->_callbackVector[i]->valueChanged(type);
                }
            }
        }
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
             << setw(30) << ""
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
