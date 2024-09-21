#include <iostream>
#include <opencv2/highgui.hpp>

#include "Detection/Model/SSDModel.h"
#include "Detection/Video/Video.h"
#include "Detection/Detection.h"

int main()
{
    auto detection = new Detection();

    detection->draw();
    return 0;
}
