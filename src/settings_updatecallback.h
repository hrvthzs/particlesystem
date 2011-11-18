#ifndef __SETTINGS_UPDATECALLBACK_H__
#define __SETTINGS_UPDATECALLBACK_H__

#include "settings.h"

namespace Settings {

    class UpdateCallback {

        public:
            virtual ~UpdateCallback() {}
            virtual void valueChanged(RecordType type) = 0;

    };

};

#endif // __SETTING_UPDATECALLBACK_H__