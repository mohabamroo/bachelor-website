angular.module('bachelor').controller('loginController', ['$scope', '$location', 'AuthService',
    function($scope, $location, AuthService) {
        $scope.logout = function() {
            AuthService.logout()
                .then(function() {
                    console.log('after logout in loginController');
                });
        }

    }
]);