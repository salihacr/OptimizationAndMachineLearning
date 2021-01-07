$(document).ready(function () {
    $(document).ajaxStart(function () {
        $("#wait").css("display", "block");
        $('#errorAlert').hide();
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

const num_of_ants_el = document.getElementById('ant_size');
const num_of_lives_el = document.getElementById('life_count');
const num_of_bees_el = document.getElementById('number_of_bees');
const map_list = ['marmara', 'icanadolu', 'karadeniz', 'doguanadolu', 'guneydogu', 'akdeniz', 'ege', 'anothercity'];
const map_city_length = [11, 13, 18, 14, 9, 8, 8, 20];

function cardChecked(element) {
    if (element.checked === true) {
        for (let i = 0; i < map_list.length; i++) {
            if (element.id === map_list[i]) {
                num_of_ants_el.value = map_city_length[i];
                num_of_lives_el.value = map_city_length[i];
                num_of_bees_el.value = map_city_length[i];
            }
        }
    }
}