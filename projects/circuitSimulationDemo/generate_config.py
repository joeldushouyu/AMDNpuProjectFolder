


import json
import math


import argparse
import os


def cpp_define(key, value):
    key = key.upper() 
    if isinstance(value, bool):
        return f"#define {key} {'1' if value else '0'}"
    elif isinstance(value, (int, float)):
        return f"#define {key} {value}"
    elif isinstance(value, str):
        return f'#define {key} "{value}"'
    else:
        raise TypeError(f"Unsupported type for key {key}: {type(value)}")

def generate_header(json_file, header_file):
    with open(json_file, "r") as f:
        config = json.load(f)

    with open(header_file, "w") as f:
        f.write("// Auto-generated config header\n")
        f.write("#ifndef CIRCUIT_CONFIG_H\n#define CIRCUIT_CONFIG_H\n\n")
        for key, value in config.items():
            f.write(cpp_define(key, value) + "\n")
        f.write("\n#endif // CIRCUIT_CONFIG_H\n")




def custom_floor(x, multiplier):
  return math.floor(x / multiplier) * multiplier

def custom_ceil(x, multiplier):
  return math.ceil(x / multiplier) * multiplier



def main(json_config_file:str, output_json_file:str, output_header_file:str ):
    try:
        with open(json_config_file, "r") as f:
            config_data = json.load(f)
            print("Successfully loaded the JSON data from 'config.json':")
            print(config_data)

            # You can now access the individual parameters like this:
            trace_size = int(config_data.get("trace_size"))

            simulate_end_time = config_data.get("simulate_end_time")
            switch_size = config_data.get("switch_size")
            diode_size = config_data.get("diode_size")
            u_size = config_data.get("u_size")
            state_size = config_data.get("state_size")
            y_size = config_data.get("y_size")

            print(f"\nTrace Size: {trace_size}")
            print(f"Simulate End Time: {simulate_end_time}")
            print(f"Switch Size: {switch_size}")
            print(f"Diode Size: {diode_size}")
            print(f"U Size: {u_size}")
            print(f"State Size: {state_size}")
            print(f"Output Size: {y_size}")
            
            total_switch_size = 2**(switch_size + diode_size)
            
            # For each diode, the matrix is      3*(diode_number) x (state_size+u_size)
            #NOTE: the host should already add (u_size) of column concat with C_diode_impulse_sw, if not memory reordering is not possible with constant stride in dma
            C1_DSW_col_size = (state_size+u_size) 
            C1_DSW_row_size =   custom_ceil(3* diode_size,16)
            switch_diode_mat_size =  C1_DSW_col_size* C1_DSW_row_size
            # Iteration matrix consist of A_with_dep, B_with_dep, C_imp, C_natual, D_imp, D_natual

            # A_with_dep and B_with_dep can be combined into one matrix of size (state_size x (state_size+u_size))
            # And C_impulse_mat, D_impulse_mat can be combine into one matrix with of size (Y_size x(state_size + u_size))
            # and C_non_impulse_matrix, D_non_impulse_matrix can be combined into another matrix of size (Y_size x (state_size+u_size))
            _A_B_C_D_mat_row =  custom_ceil( state_size + 2*y_size, 16)
            _A_B_C_D_mat_col = (state_size + u_size)
            A_B_C_D_mat_size = _A_B_C_D_mat_row * _A_B_C_D_mat_col
            
            _len_of_switch_size = (custom_ceil(switch_size+diode_size,32)//32 )   # number of 4byte(float) use for sending external switch for each iteration
            len_of_input_for_each_iteration = u_size+ _len_of_switch_size
            len_of_output_for_each_iteration = custom_ceil(y_size, 16)
            
            buffer_size_of_switch_diode = total_switch_size*switch_diode_mat_size
            buffer_A_B_C_D_size = total_switch_size * A_B_C_D_mat_size
            
            buffer_size_of_cur_X_U = custom_ceil(  state_size+ u_size , 16)
            buffer_size_of_C1_DSW_mat_res = C1_DSW_row_size
            buffer_size_of_A_B_C_D_mat_res =_A_B_C_D_mat_row
            
            
            # note: because load 16 float at a time for vector instruction, need to ensure the address are aligned to 64byte(4*16)
            buffer_size_for_in_out = ((63)*(1024))//4 - (buffer_size_of_switch_diode +buffer_A_B_C_D_size\
                + buffer_size_of_cur_X_U  + buffer_size_of_C1_DSW_mat_res + buffer_size_of_A_B_C_D_mat_res)
            # define a ping pong for it?
            
            _max_iteration_step = int(custom_floor( buffer_size_for_in_out//(len_of_input_for_each_iteration + len_of_output_for_each_iteration),2)) #TODO: round down instead?
            iteration_step_per_buffer = _max_iteration_step //2
            buffer_size_of_in_ping_pong = len_of_input_for_each_iteration*(iteration_step_per_buffer)
            buffer_size_of_out_ping_pong = len_of_output_for_each_iteration*(iteration_step_per_buffer)
            
            #TODO: for now
            ping_pong_buffer_iteration = 4
            # now write the final config result to file

            extracted_Data = {
                "trace_size": trace_size,
                "state_size": state_size,
                "u_size": u_size,
                "y_size": y_size,
                "diode_size": diode_size,
                "switch_size": switch_size,
                "C1_DSW_row_size": C1_DSW_row_size,
                "C1_DSW_col_size": C1_DSW_col_size,
                "C1_DSW_matrix_size":switch_diode_mat_size,
                "C1_DSW_buffer_size": buffer_size_of_switch_diode,
                
                "A_B_C_D_row_size": _A_B_C_D_mat_row,
                "A_B_C_D_col_size": _A_B_C_D_mat_col,
                "A_B_C_D_matrix_size": A_B_C_D_mat_size,
                "A_B_C_D_buffer_size": buffer_A_B_C_D_size,
                
                "input_switch_size": _len_of_switch_size,
                "input_size_per_iteration": len_of_input_for_each_iteration,
                "output_size_per_iteration": len_of_output_for_each_iteration,
                "iteration_step_per_ping_pong_buffer": iteration_step_per_buffer,
                "buffer_size_of_in_ping_poing": buffer_size_of_in_ping_pong,
                "buffer_size_of_out_ping_pong": buffer_size_of_out_ping_pong,
                "ping_pong_buffer_iteration": ping_pong_buffer_iteration,
                
                "total_switch_diode_state": total_switch_size,
                
                "buffer_size_of_cur_X_U": buffer_size_of_cur_X_U,
                "buffer_size_of_C1_DSW_mat_res":buffer_size_of_C1_DSW_mat_res,
                "buffer_size_of_A_B_C_D_mat_res": buffer_size_of_A_B_C_D_mat_res
                
            }            
            with open(output_json_file,"w") as outfile:
                json.dump( extracted_Data, outfile, indent=4)
            
            
            generate_header(output_json_file, output_header_file)

    except FileNotFoundError:
        print("Error: The file 'config.json' was not found.")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON in 'config.json': {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        
        






if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process circuit simulation configuration.")
    parser.add_argument("input_json", help="Path to the input configuration JSON file.")
    parser.add_argument("--final_json", default="final_config.json", help="Path to output processed JSON file.")
    parser.add_argument("--header", default="circuitConfig.h", help="Path to output C header file.")

    args = parser.parse_args()

    main(args.input_json, args.final_json, args.header)
