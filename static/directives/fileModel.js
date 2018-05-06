angular.module('bachelor')
    .directive('fileModel', ['$parse', function($parse) {
        return {
            restrict: 'A',
            link: function(scope, element, attrs) {
                var model = $parse(attrs.fileModel);
                var modelSetter = model.assign;
                element.bind('change', function() {
                    scope.$apply(function() {
                        modelSetter(scope, element[0].files[0]);
                    })
                })
            }
        }
    }])
    .directive('fileModelSub', ['$parse', function($parse) {
        return {
            restrict: 'A',
            link: function(scope, element, attrs) {
                var model2 = $parse(attrs.fileModelSub);
                console.log(attrs);
                console.log(scope.$parent);
                var modelSetter2 = model2.assign;
                element.bind('change', function() {
                    scope.$apply(function() {
                        modelSetter2(scope.$parent, element[0].files[0]);
                    })
                })
            }
        }
    }])
    .directive('onFinishRender', function($timeout) {
        return {
            restrict: 'A',
            link: function(scope, element, attr) {
                if (scope.$last === true) {
                    $timeout(function() {
                        scope.$emit(attr.onFinishRender);
                    });
                }
            }
        }
    })
    .directive("scrollRight", function() {
        return {
            link: function(scope, elem, attrs) {
                jQuery(elem).click(function() {
                    jQuery(elem).parent().siblings().animate({ scrollLeft: '+=400' }, 200, 'easeOutQuad');
                });

            }
        }
    })
    .directive("scrollLeft", function() {
        return {
            link: function(scope, elem, attrs) {
                jQuery(elem).click(function() {
                    jQuery(elem).parent().siblings().animate({ scrollLeft: '-=400' }, 200, 'easeOutQuad');
                });

            }
        }
    })
    .directive("viewSub", function() {
        return {
            link: function(scope, elem, attrs) {
                jQuery(elem).click(function() {
                    jQuery(elem).parent().siblings(".subDepartments").slideDown();
                    jQuery(elem).hide();
                    jQuery(elem).parent().children('.hideBtnSub').show();
                });

            }
        }
    })
    .directive("hideSub", function() {
        return {
            link: function(scope, elem, attrs) {
                jQuery(elem).click(function() {
                    jQuery(elem).parent().siblings(".subDepartments").slideUp();
                    jQuery(elem).hide();
                    jQuery(elem).parent().children('.viewBtnSub').show();
                });

            }
        }
    })
    .directive("ss", function() {
        return {
            link: function(scope, elem, attrs) {
                jQuery(elem).click(function() {
                    jQuery(elem).parent().siblings().animate({ scrollLeft: '-=400' }, 200, 'easeOutQuad');
                });

            }
        }
    });