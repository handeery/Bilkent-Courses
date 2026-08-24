

library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.std_logic_unsigned.all;
use IEEE.numeric_std.all;


entity pong_game is
  Port (
    clk,game_start,btn1_u,btn1_d,btn2_u,btn2_d,ps_clk,ps_data: in std_logic;
  red,green,blue: out std_logic_vector(3 downto 0);
  seg :  out std_logic_vector(6 downto 0);
   an :  out std_logic_vector(3 downto 0);
  hsync, vsync: out std_logic
  
   );
end pong_game;

architecture Behavioral of pong_game is



signal   timeout,score_p1,score_p2,col_type,col_timeout ,count_sevseg: integer:=0;
signal ven ,p1_area,p2_area,scored,collision: std_logic:='0';
signal sloww ,direction,scr1,scr2,num: std_logic_vector(1 downto 0):="00";

signal game_over : std_logic:='1';
signal new_char_came : std_logic;
signal char : std_logic_vector(7 downto 0);


----------------

signal  player_color_r : std_logic_vector(3 downto 0):="0000";
signal  player_color_g : std_logic_vector(3 downto 0):="1111";
signal  player_color_b : std_logic_vector(3 downto 0):="0000";

---------------
signal  line_color_code_r : std_logic_vector(3 downto 0):="1111";
signal  line_color_code_g : std_logic_vector(3 downto 0):="1111";
signal  line_color_code_b : std_logic_vector(3 downto 0):="1111";
----------
signal  ball_color_r : std_logic_vector(3 downto 0):="1111";
signal  ball_color_g : std_logic_vector(3 downto 0):="1111";
signal  ball_color_b : std_logic_vector(3 downto 0):="1111";

--------
signal ball_radius_sqrd : integer  :=9;--

signal ball_area : integer  ;--
signal hc : integer range 0 to 800 :=0;--hc
signal vc : integer range 0 to 520 :=0;--vc


signal ball_refresh_rate : integer  :=1500000;--5ms de 1 pixel

signal p1_h : integer  :=208;--
signal p1_v : integer  :=280;--
signal p2_h : integer  :=592;--
signal p2_v : integer  :=280;--
signal ball_h : integer  :=400;--
signal ball_v : integer  :=280;--

signal upper_limit : integer  :=180;--
signal lower_limit : integer  :=380;--
signal right_limit : integer  :=600;--
signal left_limit : integer  :=200;--
-----------------------------
component  keyboard_interface is
  Port (
  
  clk,ps_clk,ps_data: in std_logic;
    char: out std_logic_vector(7 downto 0);
    new_char_came: out std_logic
   );
end component;
begin

keyboard_interface1 : keyboard_interface Port map(
    clk => clk,
    ps_clk =>ps_clk,
    ps_data =>ps_data,
    char =>char,
    
    new_char_came =>new_char_came
  );
  
  

process(clk)
    begin
      if rising_edge(clk)then
         
         
       sloww<=sloww+ "01";
       if(sloww="11") then
          if (hc = 799) then        
              ven <= '1';              
              hc  <= 0;                   
          
          else 
              hc  <= hc + 1;          
              ven <= '0';                  
          end if;
          
      
      if (hc < 96) then            
              hsync <= '0';
          
          else  
              hsync <= '1';              
          end if;
          
          
            if (ven = '1') then         
                if (vc = 520) then      
                    vc <= 0;               
                
            else 
                vc <= vc + 1;           
            end if;
            
            if (vc < 2) then            
                vsync <= '0';
            
            else  
                vsync <= '1';           
            end if;
        end if;
      
          end if;
          


	end if;
end process ;



process(clk)
begin
    if rising_edge(clk)then
     ball_area<= (ball_h-hc)*(ball_h-hc) + (ball_v-vc)*(ball_v-vc);--ball radius =3
     if((hc >= 205) and (hc <= 211) and (vc >= p1_v-5) and (vc <= p1_v+5)) then
        p1_area<='1';
     else
        p1_area<='0';
      end if;
        
     if ( (hc>=589) and (hc<=595) and (vc>=p2_v-5) and (vc<=p2_v+5)) then
            p2_area<='1';
      else
        p2_area<='0';
      end if;
    
        if( vc=upper_limit and  hc<=right_limit and hc>=left_limit )then --upper limit
            red<=line_color_code_r;
            green<=line_color_code_g;
            blue<=line_color_code_b;            

        elsif( vc=lower_limit and  hc<=right_limit and hc>=left_limit )then --lower limit
            red<=line_color_code_r;
            green<=line_color_code_g;
            blue<=line_color_code_b;            

        elsif( hc=left_limit and  vc<=lower_limit and vc>=upper_limit )then --
            red<=line_color_code_r;
            green<=line_color_code_g;
            blue<=line_color_code_b;   

        elsif( hc=right_limit and  vc<=lower_limit and vc>=upper_limit )then --
            red<=line_color_code_r;
            green<=line_color_code_g;
            blue<=line_color_code_b;   
            
        elsif(ball_area <=ball_radius_sqrd)then--ball
            red<=ball_color_r;
            green<=ball_color_g;
            blue<=ball_color_b;                     
        elsif(p1_area='1' or p2_area='1')then--p1 and p2 
            red<=player_color_r;
            green<=player_color_g;
            blue<=player_color_b;                                                                                                                                 
        else
            red<="0000";
            green<="0000";
            blue<="0000";
        end if;
	end if;
end process ;


  --direction mapping :
  -- 
  --        0          1 
  --        
  --
  --        2         3
  --
  --
  


process(clk)
begin
    if rising_edge(clk)then
        if(score_p1=0 ) then
            scr1<="00";
        elsif(score_p1=1 ) then
            scr1<="01";
        elsif(score_p1=2 ) then
            scr1<="10";
        elsif(score_p1>=3 ) then
            scr1<="11";
        end if;

        if(score_p2=0 ) then
            scr2<="00";
        elsif(score_p2=1 ) then
            scr2<="01";
        elsif(score_p2=2 ) then
            scr2<="10";
        elsif(score_p2>=3 ) then
            scr2<="11";
        end if;
                                 
        if(count_sevseg<=300000  ) then
            num<=scr1;
            an<="0111";
            count_sevseg<=count_sevseg+1;
        elsif(count_sevseg<=600000  ) then
            an<="1110";
            num<=scr2;
            count_sevseg<=count_sevseg+1;            
        else
            count_sevseg<=0;
        end if;
        
        case num is
            when "00" => seg <= "0000001"; -- "0"     
            when "01" => seg <= "1001111"; -- "1" 
            when "10" => seg <= "0010010"; -- "2" 
            when "11" => seg <= "0000110"; -- "3"
         end case;             
                if(game_over='1') then
                    timeout<=0;

                    scored<='0';
               
                    if(game_start='1') then
                        game_over<='0';
                        direction<=sloww;--random!
                        score_p1<=0;
                        score_p2<=0;
                        p1_h<=208;
                        p1_v<=280;
                        p2_h<=592;
                        p2_v<=280;
                        ball_h<=400;
                        ball_v<=280;                                                 
                    end if;
                elsif(scored='1') then
                    timeout<=0;
                    scored<='0';
                    p1_h<=208;
                    p1_v<=280;
                    p2_h<=592;
                    p2_v<=280;
                    ball_h<=400;
                    ball_v<=280;
                    direction<=sloww;--random!
                    if(score_p1>=3 or score_p2>=3 ) then
                        game_over<='1';
                    else
                        game_over<='0';
                    end if;
                elsif(new_char_came='1') then
                    timeout<=timeout+1;
                    if(char=x"1d") then --W
                        if(p1_v>=upper_limit+3) then
                            p1_v<=p1_v-5;
                        end if;
                    elsif(char=x"1b") then --S
                        if(p1_v<=lower_limit-3) then
                            p1_v<=p1_v+5;
                        end if;

                    elsif(char=x"44") then --O
                        if(p2_v>=upper_limit+3) then
                            p2_v<=p2_v-5;
                        end if;
                    elsif(char=x"42") then --K
                        if(p2_v<=lower_limit-3) then
                            p2_v<=p2_v+5;
                        end if;
                    end if;        

                                       
                elsif(timeout>ball_refresh_rate) then --5ms = 500000
                     timeout<=0;

                     if(direction="00") then
                        ball_h<=ball_h-1;
                        ball_v<=ball_v-1;
                     elsif(direction="01") then
                        ball_h<=ball_h+1;
                        ball_v<=ball_v-1;
                     elsif(direction="10") then
                        ball_h<=ball_h-1;
                        ball_v<=ball_v+1;
                     elsif(direction="11") then
                        ball_h<=ball_h+1;
                        ball_v<=ball_v+1;
                    end if;      
                 elsif(collision='1') then
                    timeout<=timeout+1;
                    if(col_type=0) then-- ball and p1
                        direction<=direction+"01";
                    elsif(col_type=1) then-- ball and p2
                        direction<=direction-"01";
                    elsif(col_type=2) then-- ball and upper
                        direction(1)<='1';
                    elsif(col_type=3) then-- ball and lower
                        direction(1)<='0';
                    elsif(col_type=4) then-- ball and left
                        score_p2<=score_p2+1;
                        scored<='1';
                    elsif(col_type=5) then-- ball and right
                        score_p1<=score_p1+1;
                        scored<='1';
                    end if;
                                        
                 else
                      timeout<=timeout+1;
                  end if;                                        
                      
                                                                
    end if;
end process;
    



process(clk)--collusion res
begin
    if rising_edge(clk)then
        if(col_timeout<10000000) then
            col_timeout<=col_timeout+1;
            collision<='0';
        elsif(ball_area <=ball_radius_sqrd and p1_area='1') then--ball and p1
            collision<='1';
            col_type<=0;
            col_timeout<=0;
        elsif(ball_area <=ball_radius_sqrd and p2_area='1') then -- ball and p2
            collision<='1';
            col_type<=1;
            col_timeout<=0;
       elsif( ball_v<=183) then --ball upper
            collision<='1';
            col_type<=2;
            col_timeout<=0;
       elsif( ball_v>=377) then --ball lower
            collision<='1';
            col_type<=3;
            col_timeout<=0;
       elsif( ball_h<=203) then --ball left
            collision<='1';
            col_type<=4;
            col_timeout<=0;
       elsif( ball_h>=597) then --ball right
            collision<='1';
            col_type<=5;
            col_timeout<=0;    
        else
           collision<='0';                                            
        end if;
        
        
    end if;
end process;
            
end Behavioral;
