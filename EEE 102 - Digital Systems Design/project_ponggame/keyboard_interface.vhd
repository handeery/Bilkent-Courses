




library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.std_logic_unsigned.all;
use IEEE.numeric_std.all;



entity keyboard_interface is
  Port (
  
  clk,ps_clk,ps_data: in std_logic;
    char: out std_logic_vector(7 downto 0);
    new_char_came: out std_logic
   );
end keyboard_interface;

architecture Behavioral of keyboard_interface is



signal counter : std_logic_vector(7 downto 0):="00000000";
signal debug : std_logic_vector(24 downto 0);
signal ps_clk_s,ps_clk_2s,ps_data_s,ps_data_2s : std_logic;
signal state : std_logic_vector(11 downto 0):="000000000000";
signal data : std_logic_vector(10 downto 0):="00000000000";
signal bit_cnt : integer :=0;
begin


process(clk)
    begin

      if rising_edge(clk)then
        counter<=counter+"00000001";
        ps_clk_s<=ps_clk;
        ps_clk_2s<=ps_clk_s;
        
        ps_data_s<=ps_data;
        ps_data_2s<=ps_data_s;      
         case state is

            when "000000000000" =>
                new_char_came<='0';
                bit_cnt<=0;
                 state<=state+"000000000001";
                 data<="00000000000";
            when "001000000000" =>               
                if(ps_clk_2s='0') then --first negedge
                    state<=state+"000000000001";
                    data<= ps_data_2s & data(10 downto 1);
                end if;
            when "010000000000" =>
                if(ps_clk_2s='1') then --posedge
                    state<=state+"000000000001";
                    bit_cnt<=bit_cnt+1;
                end if;
             when "010000000001" =>
                if(bit_cnt>=11) then
                    state<=state+"000000000001";
                else
                  state<="000000000001";
                end if;
             when "010000000010" =>
                state<="000000000000";
                if(data(0) ='0' and data(10) ='1'    ) then
                    new_char_came<='1';
                    char <=data(8 downto 1);
                end if; 

            when others =>
                 state<=state+"000000000001";
                 new_char_came<='0';                        
         end case;
     end if;
 end process;        
         

end Behavioral;
