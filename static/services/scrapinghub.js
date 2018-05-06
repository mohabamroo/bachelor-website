angular.module('bachelor')
    .service('ScrapingService', ['$q', '$timeout', '$http', '$rootScope', 'AuthToken',
        function($q, $timeout, $http, $rootScope, AuthToken) {
            $rootScope.scrapyRoot = "https://app.scrapinghub.com/api";
            $rootScope.scrapyAPIKEY = "32f54a1220b84bbab635e6274271215a";
            $rootScope.scrapyProject = "306033";
            // return available functions for use in the controllers
            return ({
                crawlUser: crawlUser
            });

            function crawlUser(user, backend_host) {
                var deferred = $q.defer();
                var url = $rootScope.scrapyRoot + "/run.json";
                $http({
                        method: 'POST',
                        url: url,
                        params: { apikey: $rootScope.scrapyAPIKEY },
                        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
                        transformRequest: function(obj) {
                            var str = [];
                            for (var p in obj)
                                str.push(encodeURIComponent(p) + "=" + encodeURIComponent(obj[p]));
                            return str.join("&");
                        },
                        data: {
                            project: $rootScope.scrapyProject,
                            spider: 'codeforces',
                            username: user.username,
                            email: user.email,
                            backend_domain: backend_host
                        }
                    })
                    .success(function(res) {
                        deferred.resolve(res);
                    })
                    .catch(function(res) {
                        deferred.reject(res);
                    });

                return deferred.promise;
            }

        }
    ]);