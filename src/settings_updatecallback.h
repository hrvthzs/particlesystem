#ifndef __SETTINGS_UPDATECALLBACK_H__
#define __SETTINGS_UPDATECALLBACK_H__

#include "settings.h"

namespace Settings {

    /**
     * Abstract class for handling database value change notifications
     */
    class UpdateCallback {

        public:

            /**
             *  Destructor
             */
            virtual ~UpdateCallback() {}

            /**
             * Value change handler method
             *
             * @param type record type
             */
            virtual void valueChanged(RecordType type) = 0;

    };

};

#endif // __SETTING_UPDATECALLBACK_H__