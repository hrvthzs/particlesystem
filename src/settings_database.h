#ifndef __SETTINGS_DATABASE_H__
#define __SETTINGS_DATABASE_H__

#include <map>
#include <vector>
#include <string>

#include "settings.h"
#include "settings_updatecallback.h"

using namespace std;

namespace Settings {

    class Database {

        public:

            Database();
            virtual ~Database();

            void insert(
                RecordType type,
                string name,
                float minimum,
                float maximum,
                float value,
                string unit = "",
                bool editable = true
            );

            void insert(
                RecordType type,
                string name,
                float minimum,
                float maximum,
                float value,
                bool editable
            );

            void insert(
                RecordType type,
                string name,
                float minimum,
                float maximum,
                float value,
                string unit
            );

            void updateMinimum(RecordType type, float minimum);
            void updateMaximum(RecordType type, float maximum);
            void updateValue(RecordType type, float value);
            float selectValue(RecordType type);

            void addUpdateCallback(UpdateCallback* callback);

            void print();

        private:

            typedef map<RecordType, Record> SettingsMap;
            typedef vector<UpdateCallback*> CallbackVector;

            SettingsMap _settingsMap;
            CallbackVector _callbackVector;
    };

};

#endif // __SETTINGS_DATABASE_H__