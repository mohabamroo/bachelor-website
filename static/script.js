var backendUrl = "https://localhost:6003";
backendUrl = "";

angular.module('users', [], ['$routeProvider', function($routeProvider) {
    $routeProvider
        .when('/signin', {
            templateUrl: '/static/views/signin.html',
            controller: 'mainController',
            access: { restricted: false }
        })
        .when('/signup', {
            templateUrl: '/static/views/signup.html',
            controller: 'mainController',
            access: { restricted: false }
        })
        .when('/me', {
            templateUrl: '/static/views/profile.html',
            controller: 'UserController',
            access: { restricted: true }
        })
}]);

angular.module('recommendations', [], ['$routeProvider', function($routeProvider) {
    $routeProvider
        .when('/recommendations', {
            templateUrl: '/static/views/recommendations.html',
            controller: 'RecommenderController',
            access: { restricted: true }
        })
}]);
var app = angular.module('bachelor', ['ngRoute', 'users', 'recommendations']);

app.config(['$routeProvider', '$locationProvider', '$httpProvider', function($routeProvider, $locationProvider, $httpProvider) {
    // $routeProvider
    //     .when('/', {
    //         templateUrl: '/views/offers/offersIndex.html',
    //         controller: 'OffersController',
    //         access: { restricted: true }
    //     })

    $httpProvider.interceptors.push('AuthInterceptors');

    // to remove # from url, useless backend urls!
    $locationProvider.html5Mode({
        enabled: false,
        requireBase: false
    });

}]);


app.run(function($rootScope, $location, $route, $window, AuthService, UserService, AuthToken) {
    $rootScope.$watch(AuthService.isLoggedIn(), function() {
        $rootScope.userStatus = AuthService.isLoggedIn();

    });
    if (AuthService.isLoggedIn()) {
        $rootScope.current_username = AuthToken.getUsername();
    }
    $rootScope.$on('$routeChangeStart',
        function(event, next, current) {
            if (next && next.access && next.access.restricted && !AuthService.isLoggedIn()) {
                $rootScope.notAuthorized();
            }

        });

    $rootScope.$on('$viewContentLoaded', function() {
        // $rootScope.updateWidth();
        // $rootScope.updateDateInputs();
        //Here your view content is fully loaded !!
    });
});

app.directive('exit', function() {
    return function(scope, element, attrs) {
        element.bind("keydown keypress", function(event) {
            if (event.which === 27) {
                console.log("exit key");
                // scope.$apply(function (){
                //     scope.$eval(attrs.myEnter);
                // });

                event.preventDefault();
            }
        });
    };
});