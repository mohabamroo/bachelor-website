angular.module('bachelor').controller('UserController', ['$scope', '$location', 'UserService', 'AuthToken',
    '$rootScope', '$timeout', '$window', '$http', '$routeParams',
    function($scope, $location, UserService, $rootScope, $timeout, $window, $http, $routeParams, AuthToken) {


        $scope.getProfile = function() {
            $scope.overlay();
            console.log("ds;kdfjn")
            UserService.getUser()
                .then(function(response) {
                    console.log("skdsnl")
                    $scope.stopLoading();
                    $scope.user = response.data.user;
                })
                .catch(function(response) {
                    console.log("dflkdnldjfn")
                    console.log(response);
                    $scope.handleErrors(response);
                });

        }
    }
]);