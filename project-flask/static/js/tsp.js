$(document).ready(function () {
    $(document).ajaxStart(function () {
        $("#wait").css("display", "block");
    });


    $(document).ajaxComplete(function () {
        $("#wait").css("display", "none");
    });


    $('#tspform').on('submit', function (event) {
        window.scrollBy(0, 100);

        // If the plotting area is open when the button is first clicked, close it.
        $("#plot_area").css("display", "none");

        $("#btn_run").attr("disabled", true);
        $("#btn_run").css("cursor", 'wait');

        var form_data = new FormData($('#tspform')[0]);
        $.ajax({
            type: 'POST',
            url: '/solvetsp',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            enctype: 'multipart/form-data'
        })
            .done(function (data) {
                $('#errorAlert').hide();
                if (data.error) {
                    $('#errorAlertText').text(data.error);
                    $('#errorAlert').show();
                    $('#successAlert').hide();
                    $("#btn_run").attr("disabled", false);
                    $("#btn_run").css("cursor", 'pointer');
                }
                else {
                    console.log(data);
                    $('#ga_vs_aco_routes').attr("src", data.compare_routes_fig_path);
                    $('#ga_vs_aco_cost').attr("src", data.compare_costs_fig_path);
                    $('#aco_route').attr("src", data.antcolony_route_path);
                    $('#aco_cost').attr("src", data.antcolony_cost_path);
                    $('#abc_route').attr("src", data.beecolony_route_path);
                    $('#abc_cost').attr("src", data.beecolony_cost_path);
                    $('#ga_route').attr("src", data.ga_route_path);
                    $('#ga_cost').attr("src", data.ga_cost_path);

                    $("#plot_area").css("display", "block");
                    $('#errorAlert').hide();
                    $("#btn_run").attr("disabled", false);
                    $("#btn_run").css("cursor", 'pointer');
                }
            });
        event.preventDefault();
    });

});