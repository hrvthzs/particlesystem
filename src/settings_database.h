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
             * @param value actual value
             * @param unit unit type (kg, cm, etc.)
             * @param editable if true is read only
             * @return itself
             */
            Database* insert(
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
             * @param value actual value
             * @param editable if false is read only
             * @return itself
             */
            Database* insert(
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
             * @param value actual value
             * @param unit unit type (kg, cm, etc.)
             * @return itself
             */
            Database* insert(
                RecordType type,
                string name,
                float minimum,
                float maximum,
                float value,
                string unit
            );


            /**
             * Insert new record to database
             * Creates non editable record
             *
             * @param type type of record
             * @param name record name
             * @param value actual value
             * @param unit unit type (kg, cm, etc.)
             * @return itself
             */
            Database* insert(
                RecordType type,
                string name,
                float value,
                string unit = ""
            );

            /**
             * Insert new record to database
             * Creates non editable record
             *
             * @param type type of record
             * @param name record name
             * @param value actual value
             * @param editable if false is read only
             * @return itself
             */
            Database* insert(
                RecordType type,
                string name,
                float value,
                bool editable
            );

            /**
             * Update minimal value for record
             *
             * @param type type of record
             * @param minimum value
             * @return itself
             */
            Database* updateMinimum(RecordType type, float minimum);

            /**
             * Update maximal value for record
             *
             * @param type type of record
             * @param maximum value
             * @return itself
             */
            Database* updateMaximum(RecordType type, float maximum);

            /**
             * Update value of record
             *
             * @param type type of record
             * @param value value
             * @return itself
             */
            Database* updateValue(RecordType type, float value);

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