angular.module('bachelor')
    .service('UserService', ['$q', '$timeout', '$http', '$rootScope', 'AuthToken',
        function($q, $timeout, $http, $rootScope, AuthToken) {
            var user = null;

            // return available functions for use in the controllers
            return ({
                getUser: getUser
            });

            function getUser() {
                var deferred = $q.defer();
                console.log('ldkndfkln')
                if (AuthToken.getToken()) {
                    $http({
                            url: backendUrl + '/api/users/profile/',
                            method: "GET",
                            withCredentials: true
                        })
                        .then(function(response) {
                            AuthToken.setUsername(response.data.user.username);
                            $rootScope.current_username = response.data.user.username;
                            console.log(AuthToken.getUsername());
                            deferred.resolve(response);
                        }, function(response) {
                            deferred.reject(response);
                        });
                } else {
                    $q.reject({ msg: "User is not signed in." });
                }
                return deferred.promise;
            }

        }
    ]);