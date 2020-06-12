$(function () {

    $.validator.addMethod(
        "regex",
        function (value, element, regexp) {
            if (regexp.constructor != RegExp)
                regexp = new RegExp(regexp);
            else if (regexp.global)
                regexp.lastIndex = 0;
            return this.optional(element) || regexp.test(value);
        },
        "Please check your input."
    );

    $("form[name='input']").validate({
        rules: {
            count: {
                required: true,
            },
            scores: {
                required: true,
            },
            search: {
                required: true,
                url: true,
                regex: /^https?:\/\/(www.)?play.google.com\/store\/apps\/details\?id=/
            },          
        },
        messages: {
            scores: "asdasd",
            count: "asdasdas",
            search: {
                required: "Please enter a valid URL",
                regex: "Please enter a valid URL in the form of https://play.google.com/store/apps/details?id="    
            }
        },
        // tooltip_options: {
        //     search: { placement: 'left' },
        //     count: { placement: 'left' },
        //     scores: { placement: 'left' }
        //  },
        
        //   errorClass: 'validation-error-message help-block form-helper bold',
        // errorPlacement: function (error, element) {
        //     switch (element.attr("name")) {
        //         case 'search':
        //             // $('[id*=first-wrap]').text($(error).text());
        //             error.insertAfter($('[id*=first-wrap]'));
        //             break;
        //         case 'count':
        //             // $('[id*=second-wrap]').text($(error).text());
        //             error.insertAfter($('[id*=second-wrap]'));
        //             break;
        //         case 'scores':
        //             // $('[id*=third-wrap]').text($(error).text());
        //             error.insertAfter($('[id*=third-wrap]'));
        //             break;
        //     }
        // },
        submitHandler: function (form) {
            form.submit();
        }
    });
});

