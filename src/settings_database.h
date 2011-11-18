#ifndef __SETTINGS_DATABASE_H__
#define __SETTINGS_DATABASE_H__

#include <map>
#include <vector>
#include <string>

#include "settings.h"
#include "settings_updatecallback.h"

using namespace std;

namespace Settings {

    /**
     * Database class for storing simulation settings
     * On value change added callbacks are triggered
     */
    class Database {

        public:
            /*
             * Constructor
             */
            Database();

            /**
             * Destructor
             */
            virtual ~Database();

            /**
             * Insert new record to database
             *
             * @param type type of record
             * @param name record name
             * @param minimum minimal value for record
             * @param maximum maximal value for record
             * @param unit unit type (kg, cm, etc.)
             * @param editable if true is read only
             */
            void insert(
                RecordType type,
                string name,
                float minimum,
                float maximum,
                float value,
                string unit = "",
                bool editable = true
            );

            /**
             * Insert new record to database
             *
             * @param type type of record
             * @param name record name
             * @param minimum minimal value for record
             * @param maximum maximal value for record
             * @param editable if true is read only
             */
            void insert(
                RecordType type,
                string name,
                float minimum,
                float maximum,
                float value,
                bool editable
            );

            /**
             * Insert new record to database
             *
             * @param type type of record
             * @param name record name
             * @param minimum minimal value for record
             * @param maximum maximal value for record
             * @param unit unit type (kg, cm, etc.)
             */
            void insert(
                RecordType type,
                string name,
                float minimum,
                float maximum,
                float value,
                string unit
            );

            /**
             * Update minimal value for record
             *
             * @param type type of record
             * @param minimum value
             */
            void updateMinimum(RecordType type, float minimum);

            /**
             * Update maximal value for record
             *
             * @param type type of record
             * @param maximum value
             */
            void updateMaximum(RecordType type, float maximum);

            /**
             * Update value of record
             *
             * @param type type of record
             * @param value value
             */
            void updateValue(RecordType type, float value);

            /**
             * Get value of record
             *
             * @param type type of record
             */
            float selectValue(RecordType type);

            /**
             * Add update callback for value change notification
             */
            void addUpdateCallback(UpdateCallback* callback);

            /**
             * Print content of database to standart output
             */
            void print();

        private:

            typedef map<RecordType, Record> SettingsMap;
            typedef vector<UpdateCallback*> CallbackVector;

            SettingsMap _settingsMap;
            CallbackVector _callbackVector;
    };

};

#endif // __SETTINGS_DATABASE_H__