import json

dreamfusion_gallery_video_names = [
    "a_20-sided_die_made_out_of_glass.mp4",
    "a_bald_eagle_carved_out_of_wood.mp4",
    "a_banana_peeling_itself.mp4",
    "a_beagle_in_a_detective's_outfit.mp4",
    "a_beautiful_dress_made_out_of_fruit,_on_a_mannequin._Studio_lighting,_high_quality,_high_resolution.mp4",
    "a_beautiful_dress_made_out_of_garbage_bags,_on_a_mannequin._Studio_lighting,_high_quality,_high_resolution.mp4",
    "a_beautiful_rainbow_fish.mp4",
    "a_bichon_frise_wearing_academic_regalia.mp4",
    "a_blue_motorcycle.mp4",
    "a_blue_poison-dart_frog_sitting_on_a_water_lily.mp4",
    "a_brightly_colored_mushroom_growing_on_a_log.mp4",
    "a_bumblebee_sitting_on_a_pink_flower.mp4",
    "a_bunch_of_colorful_marbles_spilling_out_of_a_red_velvet_bag.mp4",
    "a_capybara_wearing_a_top_hat,_low_poly.mp4",
    "a_cat_with_a_mullet.mp4",
    "a_ceramic_lion.mp4",
    "a_ceramic_upside_down_yellow_octopus_holding_a_blue_green_ceramic_cup.mp4",
    "a_chihuahua_wearing_a_tutu.mp4",
    "a_chimpanzee_holding_a_peeled_banana.mp4",
    "a_chimpanzee_looking_through_a_telescope.mp4",
    "a_chimpanzee_stirring_a_bubbling_purple_potion_in_a_cauldron.mp4",
    "a_chimpanzee_with_a_big_grin.mp4",
    "a_completely_destroyed_car.mp4",
    "a_confused_beagle_sitting_at_a_desk_working_on_homework.mp4",
    "a_corgi_taking_a_selfie.mp4",
    "a_crab,_low_poly.mp4",
    "a_crocodile_playing_a_drum_set.mp4",
    "a_cute_steampunk_elephant.mp4",
    "a_dachsund_dressed_up_in_a_hotdog_costume.mp4",
    "a_delicious_hamburger.mp4",
    "a_dragon-cat_hybrid.mp4",
    "a_DSLR_photo_of_a_baby_dragon_drinking_boba.mp4",
    "a_DSLR_photo_of_a_baby_dragon_hatching_out_of_a_stone_egg.mp4",
    "a_DSLR_photo_of_a_baby_grand_piano_viewed_from_far_away.mp4",
    "a_DSLR_photo_of_a_bagel_filled_with_cream_cheese_and_lox.mp4",
    "a_DSLR_photo_of_a_bald_eagle.mp4",
    "a_DSLR_photo_of_a_barbecue_grill_cooking_sausages_and_burger_patties.mp4",
    "a_DSLR_photo_of_a_basil_plant.mp4",
    "a_DSLR_photo_of_a_bear_dancing_ballet.mp4",
    "a_DSLR_photo_of_a_bear_dressed_as_a_lumberjack.mp4",
    "a_DSLR_photo_of_a_bear_dressed_in_medieval_armor.mp4",
    "a_DSLR_photo_of_a_beautiful_violin_sitting_flat_on_a_table.mp4",
    "a_DSLR_photo_of_a_blue_jay_standing_on_a_large_basket_of_rainbow_macarons.mp4",
    "a_DSLR_photo_of_a_bulldozer_clearing_away_a_pile_of_snow.mp4",
    "a_DSLR_photo_of_a_bulldozer.mp4",
    "a_DSLR_photo_of_a_cake_covered_in_colorful_frosting_with_a_slice_being_taken_out,_high_resolution.mp4",
    "a_DSLR_photo_of_a_candelabra_with_many_candles_on_a_red_velvet_tablecloth.mp4",
    "a_DSLR_photo_of_a_car_made_out_of_cheese.mp4",
    "a_DSLR_photo_of_A_car_made_out_of_sushi.mp4",
    "a_DSLR_photo_of_a_car_made_out_pizza.mp4",
    "a_DSLR_photo_of_a_cat_lying_on_its_side_batting_at_a_ball_of_yarn.mp4",
    "a_DSLR_photo_of_a_cat_magician_making_a_white_dove_appear.mp4",
    "a_DSLR_photo_of_a_cat_wearing_a_bee_costume.mp4",
    "a_DSLR_photo_of_a_cat_wearing_a_lion_costume.mp4",
    "a_DSLR_photo_of_a_cauldron_full_of_gold_coins.mp4",
    "a_DSLR_photo_of_a_chimpanzee_dressed_like_Henry_VIII_king_of_England.mp4",
    "a_DSLR_photo_of_a_chimpanzee_dressed_like_Napoleon_Bonaparte.mp4",
    "a_DSLR_photo_of_a_chow_chow_puppy.mp4",
    "a_DSLR_photo_of_a_Christmas_tree_with_donuts_as_decorations.mp4",
    "a_DSLR_photo_of_a_chrome-plated_duck_with_a_golden_beak_arguing_with_an_angry_turtle_in_a_forest.mp4",
    "a_DSLR_photo_of_a_classic_Packard_car.mp4",
    "a_DSLR_photo_of_a_cocker_spaniel_wearing_a_crown.mp4",
    "a_DSLR_photo_of_a_corgi_lying_on_its_back_with_its_tongue_lolling_out.mp4",
    "a_DSLR_photo_of_a_corgi_puppy.mp4",
    "a_DSLR_photo_of_a_corgi_sneezing.mp4",
    "a_DSLR_photo_of_a_corgi_standing_up_drinking_boba.mp4",
    "a_DSLR_photo_of_a_corgi_taking_a_selfie.mp4",
    "a_DSLR_photo_of_a_corgi_wearing_a_beret_and_holding_a_baguette,_standing_up_on_two_hind_legs.mp4",
    "a_DSLR_photo_of_a_covered_wagon.mp4",
    "a_DSLR_photo_of_a_cracked_egg_with_the_yolk_spilling_out_on_a_wooden_table.mp4",
    "a_DSLR_photo_of_a_cup_full_of_pens_and_pencils.mp4",
    "a_DSLR_photo_of_a_dalmation_wearing_a_fireman's_hat.mp4",
    "a_DSLR_photo_of_a_delicious_chocolate_brownie_dessert_with_ice_cream_on_the_side.mp4",
    "a_DSLR_photo_of_a_delicious_croissant.mp4",
    "a_DSLR_photo_of_A_DMC_Delorean_car.mp4",
    "a_DSLR_photo_of_a_dog_made_out_of_salad.mp4",
    "a_DSLR_photo_of_a_drum_set_made_of_cheese.mp4",
    "a_DSLR_photo_of_a_drying_rack_covered_in_clothes.mp4",
    "a_DSLR_photo_of_aerial_view_of_a_ruined_castle.mp4",
    "a_DSLR_photo_of_a_football_helmet.mp4",
    "a_DSLR_photo_of_a_fox_holding_a_videogame_controller.mp4",
    "a_DSLR_photo_of_a_fox_taking_a_photograph_using_a_DSLR.mp4",
    "a_DSLR_photo_of_a_frazer_nash_super_sport_car.mp4",
    "a_DSLR_photo_of_a_frog_wearing_a_sweater.mp4",
    "a_DSLR_photo_of_a_ghost_eating_a_hamburger.mp4",
    "a_DSLR_photo_of_a_giant_worm_emerging_from_the_sand_in_the_middle_of_the_desert.mp4",
    "a_DSLR_photo_of_a_goose_made_out_of_gold.mp4",
    "a_DSLR_photo_of_a_green_monster_truck.mp4",
    "a_DSLR_photo_of_a_group_of_dogs_eating_pizza.mp4",
    "a_DSLR_photo_of_a_group_of_dogs_playing_poker.mp4",
    "a_DSLR_photo_of_a_gummy_bear_playing_the_saxophone.mp4",
    "a_DSLR_photo_of_a_hippo_wearing_a_sweater.mp4",
    "a_DSLR_photo_of_a_humanoid_robot_holding_a_human_brain.mp4",
    "a_DSLR_photo_of_a_humanoid_robot_playing_solitaire.mp4",
    "a_DSLR_photo_of_a_humanoid_robot_playing_the_cello.mp4",
    "a_DSLR_photo_of_a_humanoid_robot_using_a_laptop.mp4",
    "a_DSLR_photo_of_a_humanoid_robot_using_a_rolling_pin_to_roll_out_dough.mp4",
    "a_DSLR_photo_of_a_human_skull.mp4",
    "a_DSLR_photo_of_a_kitten_standing_on_top_of_a_giant_tortoise.mp4",
    "a_DSLR_photo_of_a_knight_chopping_wood.mp4",
    "a_DSLR_photo_of_a_knight_holding_a_lance_and_sitting_on_an_armored_horse.mp4",
    "a_DSLR_photo_of_a_koala_wearing_a_party_hat_and_blowing_out_birthday_candles_on_a_cake.mp4",
    "a_DSLR_photo_of_a_lemur_taking_notes_in_a_journal.mp4",
    "a_DSLR_photo_of_a_lion_reading_the_newspaper.mp4",
    "a_DSLR_photo_of_a_mandarin_duck_swimming_in_a_pond.mp4",
    "a_DSLR_photo_of_a_model_of_the_eiffel_tower_made_out_of_toothpicks.mp4",
    "a_DSLR_photo_of_a_mouse_playing_the_tuba.mp4",
    "a_DSLR_photo_of_a_mug_of_hot_chocolate_with_whipped_cream_and_marshmallows.mp4",
    "a_DSLR_photo_of_an_adorable_piglet_in_a_field.mp4",
    "a_DSLR_photo_of_an_airplane_taking_off_from_the_runway.mp4",
    "a_DSLR_photo_of_an_astronaut_standing_on_the_surface_of_mars.mp4",
    "a_DSLR_photo_of_an_eggshell_broken_in_two_with_an_adorable_chick_standing_next_to_it.mp4",
    "a_DSLR_photo_of_an_elephant_skull.mp4",
    "a_DSLR_photo_of_an_exercise_bike_in_a_well_lit_room.mp4",
    "a_DSLR_photo_of_an_extravagant_mansion,_aerial_view.mp4",
    "a_DSLR_photo_of_an_ice_cream_sundae.mp4",
    "a_DSLR_photo_of_an_iguana_holding_a_balloon.mp4",
    "a_DSLR_photo_of_an_intricate_and_complex_dish_from_a_michelin_star_restaurant.mp4",
    "a_DSLR_photo_of_An_iridescent_steampunk_patterned_millipede_with_bison_horns.mp4",
    "a_DSLR_photo_of_an_octopus_playing_the_piano.mp4",
    "a_DSLR_photo_of_an_old_car_overgrown_by_vines_and_weeds.mp4",
    "a_DSLR_photo_of_an_old_vintage_car.mp4",
    "a_DSLR_photo_of_an_orangutan_making_a_clay_bowl_on_a_throwing_wheel.mp4",
    "a_DSLR_photo_of_an_orc_forging_a_hammer_on_an_anvil.mp4",
    "a_DSLR_photo_of_an_origami_motorcycle.mp4",
    "a_DSLR_photo_of_an_ornate_silver_gravy_boat_sitting_on_a_patterned_tablecloth.mp4",
    "a_DSLR_photo_of_an_overstuffed_pastrami_sandwich.mp4",
    "a_DSLR_photo_of_an_unstable_rock_cairn_in_the_middle_of_a_stream.mp4",
    "a_DSLR_photo_of_a_pair_of_headphones_sitting_on_a_desk.mp4",
    "a_DSLR_photo_of_a_pair_of_tan_cowboy_boots,_studio_lighting,_product_photography.mp4",
    "a_DSLR_photo_of_a_peacock_on_a_surfboard.mp4",
    "a_DSLR_photo_of_a_pigeon_reading_a_book.mp4",
    "a_DSLR_photo_of_a_piglet_sitting_in_a_teacup.mp4",
    "a_DSLR_photo_of_a_pig_playing_a_drum_set.mp4",
    "a_DSLR_photo_of_a_pile_of_dice_on_a_green_tabletop_next_to_some_playing_cards.mp4",
    "a_DSLR_photo_of_a_pirate_collie_dog,_high_resolution.mp4",
    "a_DSLR_photo_of_a_plate_of_fried_chicken_and_waffles_with_maple_syrup_on_them.mp4",
    "a_DSLR_photo_of_a_plate_piled_high_with_chocolate_chip_cookies.mp4",
    "a_DSLR_photo_of_a_plush_t-rex_dinosaur_toy,_studio_lighting,_high_resolution.mp4",
    "a_DSLR_photo_of_a_plush_triceratops_toy,_studio_lighting,_high_resolution.mp4",
    "a_DSLR_photo_of_a_pomeranian_dog.mp4",
    "a_DSLR_photo_of_a_porcelain_dragon.mp4",
    "a_DSLR_photo_of_a_praying_mantis_wearing_roller_skates.mp4",
    "a_DSLR_photo_of_a_puffin_standing_on_a_rock.mp4",
    "a_DSLR_photo_of_a_pug_made_out_of_metal.mp4",
    "a_DSLR_photo_of_a_pug_wearing_a_bee_costume.mp4",
    "a_DSLR_photo_of_a_quill_and_ink_sitting_on_a_desk.mp4",
    "a_DSLR_photo_of_a_raccoon_stealing_a_pie.mp4",
    "a_DSLR_photo_of_a_red_cardinal_bird_singing.mp4",
    "a_DSLR_photo_of_a_red_convertible_car_with_the_top_down.mp4",
    "a_DSLR_photo_of_a_red-eyed_tree_frog.mp4",
    "a_DSLR_photo_of_a_red_pickup_truck_driving_across_a_stream.mp4",
    "a_DSLR_photo_of_a_red_wheelbarrow_with_a_shovel_in_it.mp4",
    "a_DSLR_photo_of_a_roast_turkey_on_a_platter.mp4",
    "a_DSLR_photo_of_a_robot_and_dinosaur_playing_chess,_high_resolution.mp4",
    "a_DSLR_photo_of_a_robot_arm_picking_up_a_colorful_block_from_a_table.mp4",
    "a_DSLR_photo_of_a_robot_cat_knocking_over_a_chess_piece_on_a_board.mp4",
    "a_DSLR_photo_of_a_robot_dinosaur.mp4",
    "a_DSLR_photo_of_a_robot_made_out_of_vegetables.mp4",
    "a_DSLR_photo_of_a_robot_stegosaurus.mp4",
    "a_DSLR_photo_of_a_robot_tiger.mp4",
    "a_DSLR_photo_of_a_rolling_pin_on_top_of_bread_dough.mp4",
    "a_DSLR_photo_of_a_sheepdog_running.mp4",
    "a_DSLR_photo_of_a_shiba_inu_playing_golf_wearing_tartan_golf_clothes_and_hat.mp4",
    "a_DSLR_photo_of_a_shiny_silver_robot_cat.mp4",
    "a_DSLR_photo_of_a_silverback_gorilla_holding_a_golden_trophy.mp4",
    "a_DSLR_photo_of_a_silver_humanoid_robot_flipping_a_coin.mp4",
    "a_DSLR_photo_of_a_small_cherry_tomato_plant_in_a_pot_with_a_few_red_tomatoes_growing_on_it.mp4",
    "a_DSLR_photo_of_a_small_saguaro_cactus_planted_in_a_clay_pot.mp4",
    "a_DSLR_photo_of_a_Space_Shuttle.mp4",
    "a_DSLR_photo_of_a_squirrel_dressed_like_a_clown.mp4",
    "a_DSLR_photo_of_a_squirrel_flying_a_biplane.mp4",
    "a_DSLR_photo_of_a_squirrel_giving_a_lecture_writing_on_a_chalkboard.mp4",
    "a_DSLR_photo_of_a_squirrel_holding_a_bowling_ball.mp4",
    "a_DSLR_photo_of_a_squirrel-lizard_hybrid.mp4",
    "a_DSLR_photo_of_a_squirrel_made_out_of_fruit.mp4",
    "a_DSLR_photo_of_a_squirrel-octopus_hybrid.mp4",
    "a_DSLR_photo_of_a_stack_of_pancakes_covered_in_maple_syrup.mp4",
    "a_DSLR_photo_of_a_steam_engine_train,_high_resolution.mp4",
    "a_DSLR_photo_of_a_steaming_basket_full_of_dumplings.mp4",
    "a_DSLR_photo_of_a_steaming_hot_plate_piled_high_with_spaghetti_and_meatballs.mp4",
    "a_DSLR_photo_of_a_steampunk_space_ship_designed_in_the_18th_century.mp4",
    "a_DSLR_photo_of_a_straw_basket_with_a_cobra_coming_out_of_it.mp4",
    "a_DSLR_photo_of_a_swan_and_its_cygnets_swimming_in_a_pond.mp4",
    "a_DSLR_photo_of_a_tarantula,_highly_detailed.mp4",
    "a_DSLR_photo_of_a_teal_moped.mp4",
    "a_DSLR_photo_of_a_teapot_shaped_like_an_elephant_head_where_its_snout_acts_as_the_spout.mp4",
    "a_DSLR_photo_of_a_teddy_bear_taking_a_selfie.mp4",
    "a_DSLR_photo_of_a_terracotta_bunny.mp4",
    "a_DSLR_photo_of_a_tiger_dressed_as_a_doctor.mp4",
    "a_DSLR_photo_of_a_tiger_made_out_of_yarn.mp4",
    "a_DSLR_photo_of_a_toilet_made_out_of_gold.mp4",
    "a_DSLR_photo_of_a_toy_robot.mp4",
    "a_DSLR_photo_of_a_train_engine_made_out_of_clay.mp4",
    "a_DSLR_photo_of_a_tray_of_Sushi_containing_pugs.mp4",
    "a_DSLR_photo_of_a_tree_stump_with_an_axe_buried_in_it.mp4",
    "a_DSLR_photo_of_a_turtle_standing_on_its_hind_legs,_wearing_a_top_hat_and_holding_a_cane.mp4",
    "a_DSLR_photo_of_a_very_beautiful_small_organic_sculpture_made_of_fine_clockwork_and_gears_with_tiny_ruby_bearings,_very_intricate,_caved,_curved._Studio_lighting,_High_resolution,_white_background.mp4",
    "a_DSLR_photo_of_A_very_beautiful_tiny_human_heart_organic_sculpture_made_of_copper_wire_and_threaded_pipes,_very_intricate,_curved,_Studio_lighting,_high_resolution.mp4",
    "a_DSLR_photo_of_a_very_cool_and_trendy_pair_of_sneakers,_studio_lighting.mp4",
    "a_DSLR_photo_of_a_vintage_record_player.mp4",
    "a_DSLR_photo_of_a_wine_bottle_and_full_wine_glass_on_a_chessboard.mp4",
    "a_DSLR_photo_of_a_wooden_desk_and_chair_from_an_elementary_school.mp4",
    "a_DSLR_photo_of_a_yorkie_dog_eating_a_donut.mp4",
    "a_DSLR_photo_of_a_yorkie_dog_wearing_extremely_cool_sneakers.mp4",
    "a_DSLR_photo_of_baby_elephant_jumping_on_a_trampoline.mp4",
    "a_DSLR_photo_of_cat_wearing_virtual_reality_headset_in_renaissance_oil_painting_high_detail_caravaggio.mp4",
    "a_DSLR_photo_of_edible_typewriter_made_out_of_vegetables.mp4",
    "a_DSLR_photo_of_Mont_Saint-Michel,_France,_aerial_view.mp4",
    "a_DSLR_photo_of_Mount_Fuji,_aerial_view.mp4",
    "a_DSLR_photo_of_Neuschwanstein_Castle,_aerial_view.mp4",
    "A_DSLR_photo_of___pyramid_shaped_burrito_with_a_slice_cut_out_of_it.mp4",
    "a_DSLR_photo_of_the_Imperial_State_Crown_of_England.mp4",
    "a_DSLR_photo_of_the_leaning_tower_of_Pisa,_aerial_view.mp4",
    "a_DSLR_photo_of_the_Statue_of_Liberty,_aerial_view.mp4",
    "a_DSLR_photo_of_Two_locomotives_playing_tug_of_war.mp4",
    "a_DSLR_photo_of_two_macaw_parrots_sharing_a_milkshake_with_two_straws.mp4",
    "a_DSLR_photo_of_Westminster_Abbey,_aerial_view.mp4",
    "a_ficus_planted_in_a_pot.mp4",
    "a_flower_made_out_of_metal.mp4",
    "a_fluffy_cat_lying_on_its_back_in_a_patch_of_sunlight.mp4",
    "a_fox_and_a_hare_tangoing_together.mp4",
    "a_fox_holding_a_videogame_controller.mp4",
    "a_fox_playing_the_cello.mp4",
    "a_frazer_nash_super_sport_car.mp4",
    "a_freshly_baked_loaf_of_sourdough_bread_on_a_cutting_board.mp4",
    "a_goat_drinking_beer.mp4",
    "a_golden_goblet,_low_poly.mp4",
    "a_green_dragon_breathing_fire.mp4",
    "a_green_tractor_farming_corn_fields.mp4",
    "a_highland_cow.mp4",
    "a_hotdog_in_a_tutu_skirt.mp4",
    "a_humanoid_robot_laying_on_the_couch_while_on_a_laptop.mp4",
    "a_humanoid_robot_playing_the_violin.mp4",
    "a_humanoid_robot_sitting_looking_at_a_Go_board_with_some_pieces_on_it.mp4",
    "a_human_skeleton_drinking_a_glass_of_red_wine.mp4",
    "a_human_skull_with_a_vine_growing_through_one_of_the_eye_sockets.mp4",
    "a_kitten_looking_at_a_goldfish_in_a_bowl.mp4",
    "a_lemur_drinking_boba.mp4",
    "a_lemur_taking_notes_in_a_journal.mp4",
    "a_lionfish.mp4",
    "a_llama_wearing_a_suit.mp4",
    "a_marble_bust_of_a_mouse.mp4",
    "a_metal_sculpture_of_a_lion's_head,_highly_detailed.mp4",
    "a_mojito_in_a_beach_chair.mp4",
    "a_monkey-rabbit_hybrid.mp4",
    "an_airplane_made_out_of_wood.mp4",
    "an_amigurumi_bulldozer.mp4",
    "An_anthropomorphic_tomato_eating_another_tomato.mp4",
    "an_astronaut_playing_the_violin.mp4",
    "an_astronaut_riding_a_kangaroo.mp4",
    "an_English_castle,_aerial_view.mp4",
    "an_erupting_volcano,_aerial_view.mp4",
    "a_nest_with_a_few_white_eggs_and_one_golden_egg.mp4",
    "an_exercise_bike.mp4",
    "an_iridescent_metal_scorpion.mp4",
    "An_octopus_and_a_giraffe_having_cheesecake.mp4",
    "an_octopus_playing_the_harp.mp4",
    "an_old_vintage_car.mp4",
    "an_opulent_couch_from_the_palace_of_Versailles.mp4",
    "an_orange_road_bike.mp4",
    "an_orangutan_holding_a_paint_palette_in_one_hand_and_a_paintbrush_in_the_other.mp4",
    "an_orangutan_playing_accordion_with_its_hands_spread_wide.mp4",
    "an_orangutan_using_chopsticks_to_eat_ramen.mp4",
    "an_orchid_flower_planted_in_a_clay_pot.mp4",
    "a_palm_tree,_low_poly_3d_model.mp4",
    "a_panda_rowing_a_boat_in_a_pond.mp4",
    "a_panda_wearing_a_necktie_and_sitting_in_an_office_chair.mp4",
    "A_Panther_De_Ville_car.mp4",
    "a_pig_wearing_a_backpack.mp4",
    "a_plate_of_delicious_tacos.mp4",
    "a_plush_dragon_toy.mp4",
    "a_plush_toy_of_a_corgi_nurse.mp4",
    "a_rabbit,_animated_movie_character,_high_detail_3d_model.mp4",
    "a_rabbit_cutting_grass_with_a_lawnmower.mp4",
    "a_red_eyed_tree_frog,_low_poly.mp4",
    "a_red_panda.mp4",
    "a_ripe_strawberry.mp4",
    "a_roulette_wheel.mp4",
    "a_shiny_red_stand_mixer.mp4",
    "a_silver_platter_piled_high_with_fruits.mp4",
    "a_sliced_loaf_of_fresh_bread.mp4",
    "a_snail_on_a_leaf.mp4",
    "a_spanish_galleon_sailing_on_the_open_sea.mp4",
    "a_squirrel_dressed_like_Henry_VIII_king_of_England.mp4",
    "a_squirrel_gesturing_in_front_of_an_easel_showing_colorful_pie_charts.mp4",
    "a_squirrel_wearing_a_tuxedo_and_holding_a_conductor's_baton.mp4",
    "a_team_of_butterflies_playing_soccer_on_a_field.mp4",
    "a_teddy_bear_pushing_a_shopping_cart_full_of_fruits_and_vegetables.mp4",
    "a_tiger_dressed_as_a_military_general.mp4",
    "a_tiger_karate_master.mp4",
    "a_tiger_playing_the_violin.mp4",
    "a_tiger_waiter_at_a_fancy_restaurant.mp4",
    "a_tiger_wearing_a_tuxedo.mp4",
    "a_t-rex_roaring_up_into_the_air.mp4",
    "a_turtle_standing_on_its_hind_legs,_wearing_a_top_hat_and_holding_a_cane.mp4",
    "a_typewriter.mp4",
    "a_walrus_smoking_a_pipe.mp4",
    "a_wedge_of_cheese_on_a_silver_platter.mp4",
    "a_wide_angle_DSLR_photo_of_a_colorful_rooster.mp4",
    "a_wide_angle_DSLR_photo_of_a_humanoid_banana_sitting_at_a_desk_doing_homework.mp4",
    "a_wide_angle_DSLR_photo_of_a_mythical_troll_stirring_a_cauldron.mp4",
    "a_wide_angle_DSLR_photo_of_a_squirrel_in_samurai_armor_wielding_a_katana.mp4",
    "a_wide_angle_zoomed_out_DSLR_photo_of_A_red_dragon_dressed_in_a_tuxedo_and_playing_chess._The_chess_pieces_are_fashioned_after_robots.mp4",
    "a_wide_angle_zoomed_out_DSLR_photo_of_a_skiing_penguin_wearing_a_puffy_jacket.mp4",
    "a_wide_angle_zoomed_out_DSLR_photo_of_zoomed_out_view_of_Tower_Bridge_made_out_of_gingerbread_and_candy.mp4",
    "a_woolly_mammoth_standing_on_ice.mp4",
    "a_yellow_schoolbus.mp4",
    "a_zoomed_out_DSLR_photo_of_a_3d_model_of_an_adorable_cottage_with_a_thatched_roof.mp4",
    "a_zoomed_out_DSLR_photo_of_a_baby_bunny_sitting_on_top_of_a_stack_of_pancakes.mp4",
    "a_zoomed_out_DSLR_photo_of_a_baby_dragon.mp4",
    "a_zoomed_out_DSLR_photo_of_a_baby_monkey_riding_on_a_pig.mp4",
    "a_zoomed_out_DSLR_photo_of_a_badger_wearing_a_party_hat_and_blowing_out_birthday_candles_on_a_cake.mp4",
    "a_zoomed_out_DSLR_photo_of_a_beagle_eating_a_donut.mp4",
    "a_zoomed_out_DSLR_photo_of_a_bear_playing_electric_bass.mp4",
    "a_zoomed_out_DSLR_photo_of_a_beautifully_carved_wooden_knight_chess_piece.mp4",
    "a_zoomed_out_DSLR_photo_of_a_beautiful_suit_made_out_of_moss,_on_a_mannequin._Studio_lighting,_high_quality,_high_resolution.mp4",
    "a_zoomed_out_DSLR_photo_of_a_blue_lobster.mp4",
    "a_zoomed_out_DSLR_photo_of_a_blue_tulip.mp4",
    "a_zoomed_out_DSLR_photo_of_a_bowl_of_cereal_and_milk_with_a_spoon_in_it.mp4",
    "a_zoomed_out_DSLR_photo_of_a_brain_in_a_jar.mp4",
    "a_zoomed_out_DSLR_photo_of_a_bulldozer_made_out_of_toy_bricks.mp4",
    "a_zoomed_out_DSLR_photo_of_a_cake_in_the_shape_of_a_train.mp4",
    "a_zoomed_out_DSLR_photo_of_a_chihuahua_lying_in_a_pool_ring.mp4",
    "a_zoomed_out_DSLR_photo_of_a_chimpanzee_dressed_as_a_football_player.mp4",
    "a_zoomed_out_DSLR_photo_of_a_chimpanzee_holding_a_cup_of_hot_coffee.mp4",
    "a_zoomed_out_DSLR_photo_of_a_chimpanzee_wearing_headphones.mp4",
    "a_zoomed_out_DSLR_photo_of_a_colorful_camping_tent_in_a_patch_of_grass.mp4",
    "a_zoomed_out_DSLR_photo_of_a_complex_movement_from_an_expensive_watch_with_many_shiny_gears,_sitting_on_a_table.mp4",
    "a_zoomed_out_DSLR_photo_of_a_construction_excavator.mp4",
    "a_zoomed_out_DSLR_photo_of_a_corgi_wearing_a_top_hat.mp4",
    "a_zoomed_out_DSLR_photo_of_a_corn_cob_and_a_banana_playing_poker.mp4",
    "a_zoomed_out_DSLR_photo_of_a_dachsund_riding_a_unicycle.mp4",
    "a_zoomed_out_DSLR_photo_of_a_dachsund_wearing_a_boater_hat.mp4",
    "a_zoomed_out_DSLR_photo_of_a_few_pool_balls_sitting_on_a_pool_table.mp4",
    "a_zoomed_out_DSLR_photo_of_a_fox_working_on_a_jigsaw_puzzle.mp4",
    "a_zoomed_out_DSLR_photo_of_a_fresh_cinnamon_roll_covered_in_glaze.mp4",
    "a_zoomed_out_DSLR_photo_of_a_green_tractor.mp4",
    "a_zoomed_out_DSLR_photo_of_a_greyhound_dog_racing_down_the_track.mp4",
    "a_zoomed_out_DSLR_photo_of_a_group_of_squirrels_rowing_crew.mp4",
    "a_zoomed_out_DSLR_photo_of_a_gummy_bear_driving_a_convertible.mp4",
    "a_zoomed_out_DSLR_photo_of_a_hermit_crab_with_a_colorful_shell.mp4",
    "a_zoomed_out_DSLR_photo_of_a_hippo_biting_through_a_watermelon.mp4",
    "a_zoomed_out_DSLR_photo_of_a_hippo_made_out_of_chocolate.mp4",
    "a_zoomed_out_DSLR_photo_of_a_humanoid_robot_lying_on_a_couch_using_a_laptop.mp4",
    "a_zoomed_out_DSLR_photo_of_a_humanoid_robot_sitting_on_a_chair_drinking_a_cup_of_coffee.mp4",
    "a_zoomed_out_DSLR_photo_of_a_human_skeleton_relaxing_in_a_lounge_chair.mp4",
    "a_zoomed_out_DSLR_photo_of_a_kangaroo_sitting_on_a_bench_playing_the_accordion.mp4",
    "a_zoomed_out_DSLR_photo_of_a_kingfisher_bird.mp4",
    "a_zoomed_out_DSLR_photo_of_a_ladybug.mp4",
    "a_zoomed_out_DSLR_photo_of_a_lion's_mane_jellyfish.mp4",
    "a_zoomed_out_DSLR_photo_of_a_lobster_playing_the_saxophone.mp4",
    "a_zoomed_out_DSLR_photo_of_a_majestic_sailboat.mp4",
    "a_zoomed_out_DSLR_photo_of_a_marble_bust_of_a_cat,_a_real_mouse_is_sitting_on_its_head.mp4",
    "a_zoomed_out_DSLR_photo_of_a_marble_bust_of_a_fox_head.mp4",
    "a_zoomed_out_DSLR_photo_of_a_model_of_a_house_in_Tudor_style.mp4",
    "a_zoomed_out_DSLR_photo_of_a_monkey-rabbit_hybrid.mp4",
    "a_zoomed_out_DSLR_photo_of_a_monkey_riding_a_bike.mp4",
    "a_zoomed_out_DSLR_photo_of_a_mountain_goat_standing_on_a_boulder.mp4",
    "a_zoomed_out_DSLR_photo_of_a_mouse_holding_a_candlestick.mp4",
    "a_zoomed_out_DSLR_photo_of_an_adorable_kitten_lying_next_to_a_flower.mp4",
    "a_zoomed_out_DSLR_photo_of_an_all-utility_vehicle_driving_across_a_stream.mp4",
    "a_zoomed_out_DSLR_photo_of_an_amigurumi_motorcycle.mp4",
    "a_zoomed_out_DSLR_photo_of_an_astronaut_chopping_vegetables_in_a_sunlit_kitchen.mp4",
    "a_zoomed_out_DSLR_photo_of_an_egg_cracked_open_with_a_newborn_chick_hatching_out_of_it.mp4",
    "a_zoomed_out_DSLR_photo_of_an_expensive_office_chair.mp4",
    "a_zoomed_out_DSLR_photo_of_an_origami_bulldozer_sitting_on_the_ground.mp4",
    "a_zoomed_out_DSLR_photo_of_an_origami_crane.mp4",
    "a_zoomed_out_DSLR_photo_of_an_origami_hippo_in_a_river.mp4",
    "a_zoomed_out_DSLR_photo_of_an_otter_lying_on_its_back_in_the_water_holding_a_flower.mp4",
    "a_zoomed_out_DSLR_photo_of_a_pair_of_floating_chopsticks_picking_up_noodles_out_of_a_bowl_of_ramen.mp4",
    "a_zoomed_out_DSLR_photo_of_a_panda_throwing_wads_of_cash_into_the_air.mp4",
    "a_zoomed_out_DSLR_photo_of_a_panda_wearing_a_chef's_hat_and_kneading_bread_dough_on_a_countertop.mp4",
    "a_zoomed_out_DSLR_photo_of_a_pigeon_standing_on_a_manhole_cover.mp4",
    "a_zoomed_out_DSLR_photo_of_a_pig_playing_the_saxophone.mp4",
    "a_zoomed_out_DSLR_photo_of_a_pile_of_dice_on_a_green_tabletop.mp4",
    "a_zoomed_out_DSLR_photo_of_a_pita_bread_full_of_hummus_and_falafel_and_vegetables.mp4",
    "a_zoomed_out_DSLR_photo_of_a_pug_made_out_of_modeling_clay.mp4",
    "a_zoomed_out_DSLR_photo_of_A_punk_rock_squirrel_in_a_studded_leather_jacket_shouting_into_a_microphone_while_standing_on_a_stump_and_holding_a_beer.mp4",
    "a_zoomed_out_DSLR_photo_of_a_rabbit_cutting_grass_with_a_lawnmower.mp4",
    "a_zoomed_out_DSLR_photo_of_a_rabbit_digging_a_hole_with_a_shovel.mp4",
    "a_zoomed_out_DSLR_photo_of_a_raccoon_astronaut_holding_his_helmet.mp4",
    "a_zoomed_out_DSLR_photo_of_a_rainforest_bird_mating_ritual_dance.mp4",
    "a_zoomed_out_DSLR_photo_of_a_recliner_chair.mp4",
    "a_zoomed_out_DSLR_photo_of_a_red_rotary_telephone.mp4",
    "a_zoomed_out_DSLR_photo_of_a_robot_couple_fine_dining.mp4",
    "a_zoomed_out_DSLR_photo_of_a_rotary_telephone_carved_out_of_wood.mp4",
    "a_zoomed_out_DSLR_photo_of_a_shiny_beetle.mp4",
    "a_zoomed_out_DSLR_photo_of_a_silver_candelabra_sitting_on_a_red_velvet_tablecloth,_only_one_candle_is_lit.mp4",
    "a_zoomed_out_DSLR_photo_of_a_squirrel_DJing.mp4",
    "a_zoomed_out_DSLR_photo_of_a_squirrel_dressed_up_like_a_Victorian_woman.mp4",
    "a_zoomed_out_DSLR_photo_of_a_table_with_dim_sum_on_it.mp4",
    "a_zoomed_out_DSLR_photo_of_a_tiger_dressed_as_a_maid.mp4",
    "a_zoomed_out_DSLR_photo_of_a_tiger_dressed_as_a_military_general.mp4",
    "a_zoomed_out_DSLR_photo_of_a_tiger_eating_an_ice_cream_cone.mp4",
    "a_zoomed_out_DSLR_photo_of_a_tiger_wearing_sunglasses_and_a_leather_jacket,_riding_a_motorcycle.mp4",
    "a_zoomed_out_DSLR_photo_of_a_toad_catching_a_fly_with_its_tongue.mp4",
    "a_zoomed_out_DSLR_photo_of_a_wizard_raccoon_casting_a_spell.mp4",
    "a_zoomed_out_DSLR_photo_of_a_yorkie_dog_dressed_as_a_maid.mp4",
    "a_zoomed_out_DSLR_photo_of_cats_wearing_eyeglasses.mp4",
    "a_zoomed_out_DSLR_photo_of_miniature_schnauzer_wooden_sculpture,_high_quality_studio_photo.mp4",
    "A_zoomed_out_DSLR_photo_of___phoenix_made_of_splashing_water_.mp4",
    "a_zoomed_out_DSLR_photo_of_Sydney_opera_house,_aerial_view.mp4",
    "a_zoomed_out_DSLR_photo_of_two_foxes_tango_dancing.mp4",
    "a_zoomed_out_DSLR_photo_of_two_raccoons_playing_poker.mp4",
    "Chichen_Itza,_aerial_view.mp4",
    "__Coffee_cup_with_many_holes.mp4",
    "fries_and_a_hamburger.mp4",
    "__Luminescent_wild_horses.mp4",
    "Michelangelo_style_statue_of_an_astronaut.mp4",
    "Michelangelo_style_statue_of_dog_reading_news_on_a_cellphone.mp4",
    "the_titanic,_aerial_view.mp4",
    "two_gummy_bears_playing_dominoes.mp4",
    "two_macaw_parrots_playing_chess.mp4",
    "Wedding_dress_made_of_tentacles.mp4",
]


def main():
    prompt_library = {
        "dreamfusion": [
            p.replace(".mp4", "").replace("_", " ")
            for p in dreamfusion_gallery_video_names
        ]
    }
    with open("av3d/threestudio/load/prompt_library.json", "w") as f:
        json.dump(prompt_library, f, indent=2)


if __name__ == "__main__":
    main()
