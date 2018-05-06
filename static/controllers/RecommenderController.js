angular.module('bachelor').controller('RecommenderController', ['$scope', '$location', 'UserService', 'AuthToken',
    '$rootScope', '$timeout', '$window', '$http', '$routeParams',
    function($scope, $location, UserService, AuthToken, $rootScope, $timeout, $window, $http, $routeParams) {
        $rootScope.scrapyRoot = "https://app.scrapinghub.com/api";
        $rootScope.scrapyAPIKEY = "32f54a1220b84bbab635e6274271215a";
        $rootScope.scrapyProject = "306033";

        $scope.codeforcesRoot = "http://codeforces.com/";

        $rootScope.getRecommendations = function() {
            $rootScope.overlay();
            $http({
                    url: backendUrl + '/recommender/recommendations/',
                    method: "get"
                })
                .then(function(response) {
                        $scope.problems = response.data.problems;
                        $rootScope.stopLoading();
                    },
                    function(response) {
                        $rootScope.handleErrors(response);
                    });
        }


        $rootScope.createJob = function(username = null) {
            $scope.overlay();
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
                        username: username
                    }
                })
                .success(function(res) {
                    $scope.stopLoading();
                    console.log(res);
                })
                .catch(function(res) {
                    console.log(res);
                    // $scope.handleErrors(res);
                });
        }
    }
]);