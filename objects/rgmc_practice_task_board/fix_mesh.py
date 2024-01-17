import subprocess

def compute_normals_meshlab(input_mesh, output_mesh):
#     script_content = """<!DOCTYPE FilterScript>
# <FilterScript>

# <filter name="Compute Normals for Point Sets" />
# <filter name="Consistent Vertex Order" />

# </FilterScript>
# """

    # Save the script to a temporary file
    script_path = '/home/vims/git/dex-mani-mujoco/objects/rgmc_practice_task_board/fix_mesh.mlx'
    # with open(script_path, 'w') as script_file:
    #     script_file.write(script_content)

    # Run meshlabserver command
    subprocess.run(['meshlabserver', '-i', input_mesh, '-o', output_mesh, '-s', script_path])

    # Remove the temporary script file
    # subprocess.run(['rm', script_path])

def main():
        #   <mesh name="task_board_68" file="../rgmc_practice_task_board/obj/ICRA2022_Practice - elastic retainer-1.obj" scale="0.001 0.001 0.001" />

    output_mesh = 'output_mesh.obj'
    input_mesh = 'obj/ICRA2022_Practice - elastic retainer-1.obj'

    compute_normals_meshlab(input_mesh, output_mesh)

if __name__ == "__main__":
    main()
