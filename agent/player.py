from .planner import Planner, save_model, load_model
import torch
import torchvision.transforms.functional as TF
import numpy as np


class HockeyPlayer:
    """
       Your ice hockey player. You may do whatever you want here. There are three rules:
        1. no calls to the pystk library (your code will not run on the tournament system if you do)
        2. There needs to be a deep network somewhere in the loop
        3. You code must run in 100 ms / frame on a standard desktop CPU (no for testing GPU)
        
        Try to minimize library dependencies, nothing that does not install through pip on linux.
    """
    
    """
       You may request to play with a different kart.
       Call `python3 -c "import pystk; pystk.init(pystk.GraphicsConfig.ld()); print(pystk.list_karts())"` to see all values.
    """
    kart = ""
    
    def __init__(self, player_id = 0):
        """
        Set up a soccer player.
        The player_id starts at 0 and increases by one for each player added. You can use the player id to figure out your team (player_id % 2), or assign different roles to different agents.
        """
        all_players = ['adiumy', 'amanda', 'beastie', 'emule', 'gavroche', 'gnu', 'hexley', 'kiki', 'konqi',
                       'nolok', 'pidgin', 'puffy', 'sara_the_racer', 'sara_the_wizard', 'suzanne', 'tux', 'wilber', 'xue']
        self.kart = all_players[np.random.choice(len(all_players))]
        self.previous_location = np.int32([0,0])

        # LOAD PLANNER MODEL FOR PUCK DETECTION
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = load_model()
        self.model.eval()

        # RECOVER PLAYER
        self.do_unstuck = False
        self.mitigation_epoch = 0
        self.mitigation_turn = -1

        self.win = ['Call me Lionel!', 'Mine', 'Out of my way!', 'You need more training.', 'I have more layers.']

    def player_orientation(self, player_front, player_location):
        facing_forward = player_front[1] > player_location[1]
        nonzero = 1e-6
        denom = player_location[0] - player_front[0]
        central = 65

        with np.errstate(divide='ignore'):
            slope = (player_location[1] - player_front[1])/(denom + nonzero)
            intersect = player_location[1] - (slope*player_location[0])
            intercept = -1 if not facing_forward and slope != 0 else 1
            intercept *= (central-intersect) / slope if slope != 0 else player_location[1]

        return (facing_forward, intercept)

    def curve_player(self, curve, towards_goal, with_opposition, focus_puck):
        # NEVER LETTING IT OUT OF YOUR SIGHT
        far_left, far_right = -40 < focus_puck < -10, 10 < focus_puck < 40
        close_left, close_right = 0 < focus_puck <= 10, -10 <= focus_puck < 0
        trajectory_bias = 2

        # WHEN FACING GOAL CLOSE IN
        if (far_left or far_right) and towards_goal:
            if (far_left and with_opposition) or (far_right and not with_opposition):
                curve -= trajectory_bias
            if (far_right and with_opposition) or (far_left and not with_opposition):
                curve += trajectory_bias

        # WHEN FACING AWAY GO WIDE
        if (close_left or close_right) and not towards_goal:
            if (close_right and with_opposition) or (close_left and not with_opposition):
                curve -= trajectory_bias
            if (close_left and with_opposition) or (close_right and not with_opposition):
                curve += trajectory_bias

        return curve

    def control(self, aim_point, player):
        # ACTION RESET
        action = {'acceleration': 1, 'brake': False, 'drift': False, 'nitro': False, 'rescue': False, 'steer': 0}

        # PLAYER WORLD LOCATION PARAMETERS
        player_front = np.float32([player.kart.front[0], player.kart.front[2]])
        player_location = np.float32([player.kart.location[0], player.kart.location[2]])
        [player_x, player_y] = player_location

        # FRAME PARAMETERS
        width, height = 400, 300
        x_center, y_center = width/2, height/2
        dead_zone = 2

        # ARENA CONDITIONS
        within_arena_center = -8 < player_x < 1 and -55 < player_y < 55

        # PLAYER STATE
        player_speed = np.sqrt(player.kart.velocity[0]**2 + player.kart.velocity[2]**2)
        player_fast = player_speed > 10
        player_very_fast = player_speed > 20
        player_slow = player_speed < 5
        player_stuck = self.previous_location[0] == np.int32(player_location)[0] and self.previous_location[1] == np.int32(player_location)[1]

        # RESOLVE PUCK LOCATION
        x_puck = aim_point[0]     
        y_puck = aim_point[1]

        # COORDINATE CLIPPING
        if x_puck > width:
            x_puck = width
        if x_puck < 0:
            x_puck = 0

        if y_puck > height:
            y_puck = height
        if y_puck < 0:
            y_puck = 0

        # WHO HAS THE PUCK?
        with_opposition = -60 < player_y < -50

        phrase = "$@!" if with_opposition else 'Superior bot: ' + np.random.choice(self.win, replace=False)
        print(phrase)

        # TRAJECTORY TO PUCK
        towards_goal, intercept_puck = self.player_orientation(player_front, player_location)

        # PLAYER CENTERED ON MAP
        player_centered = -10 < player_x < 10
        x_puck = x_puck if player_centered else self.curve_player(x_puck, towards_goal, with_opposition, intercept_puck)
        
        # PUCK INFLUENCES
        steer_left = x_puck < (x_center - dead_zone)
        steer_right = x_puck > (x_center + dead_zone)
        puck_getting_closer = x_puck < 100 or x_puck > 300
        puck_at_edge = x_puck < 50 or x_puck > 350
        puck_far = y_puck > y_center

        # ADJUSTING PLAYER DIRECTION STATE
        if steer_left:
            action['steer'] = -1
        if steer_right:
            action['steer'] = 1

        # ADJUSTING PLAYER MOTION STATE RELATIVE TO PUCK
        # BRAKE IF STUCK
        action['brake'] = True if self.do_unstuck else False
        if player_very_fast:
            action['acceleration'] = 0.2

        # SLIDE PLAYER
        if puck_at_edge:
            action['drift'] = True
            action['acceleration'] = 0.3
        else:
            action['drift'] = False

        # AVOID BEING TACKLED, SWOOP
        if puck_getting_closer:
            action['acceleration'] = 0.6

        # COUNTERMEASURE 
        if self.do_unstuck:
            action['acceleration'] = 0
            action['steer'] = self.mitigation_turn
            self.mitigation_epoch -= 3

            initial_condition = (player_slow and within_arena_center) or self.mitigation_epoch < 3

            if initial_condition:
                self.mitigation_epoch = 0
                self.do_unstuck = False
        
        else:
            if player_stuck:
                self.mitigation_epoch += 6
            
            # PLAYER NO LONGER STUCK
            else:
                if not self.do_unstuck:
                    self.mitigation_epoch = 0
            
            # PLAYER ALMOST FREE
            if self.mitigation_epoch < 2:
                self.mitigation_turn = 1 if steer_left else -1 # Do the opposite to get out
            
            # REQUIRES ADDITIONAL RECOVERY EFFORTS
            if self.mitigation_epoch > 36 or puck_far:
                if player_fast:
                    self.mitigation_epoch = 36
                    self.mitigation_turn = 0
                else:
                    self.mitigation_epoch = 18
                self.do_unstuck = True

        self.previous_location = np.int32(player_location)
        
        return action

    def act(self, image, player_info, state=None):
        """
        Set the action given the current image
        :param image: numpy array of shape (300, 400, 3)
        :param player_info: pystk.Player object for the current kart.
        return: Dict describing the action
        """
      
        """
        Your code here.
        """

        cv_puck_location = self.model(TF.to_tensor(image)[None]).squeeze(0)
        cv_puck_location = cv_puck_location.detach().cpu().numpy()

        action = self.control(cv_puck_location, player_info)

        return action
