###
    # @Date: 2023-03-07 12:08:43
 # @LastEditTime: 2023-03-17 13:17:23
    # @Description: 
    #   Copyright (c) 2022-2023 Safebench Team
    #   This work is licensed under the terms of the MIT license.
    #   For a copy, see <https://opensource.org/licenses/MIT>
### 

# start docker
docker run -it --network="host" -e CARLAVIZ_BACKEND_HOST=localhost -e CARLA_SERVER_HOST=localhost -e CARLA_SERVER_PORT=2000 mjxu96/carlaviz:0.9.13 # based on your carla version

# then open your browser and go to localhost:8080 or CARLAVIZ_BACKEND_HOST:8080
