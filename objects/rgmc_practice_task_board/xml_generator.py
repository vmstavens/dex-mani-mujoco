import os

obj_list = os.listdir("obj/")

# obj_list = os.listdir("obj/")[:67]

file_str = """
<mujocoinclude>
    <asset>
"""
for i, obj in enumerate(obj_list):
    file_str += f"      <mesh name=\"task_board_{i}\" file=\"../rgmc_practice_task_board/obj/{obj}\" scale=\"0.001 0.001 0.001\" />\n"

file_str += "   </asset>\n"

file_str += "   <worldbody>\n"

for i, obj in enumerate(obj_list):
    file_str += f"      <body name=\"task_board_{i}\" pos=\"{i/10} 0.0 0.0\" quat=\"1 0 0 1\"> \n"
    file_str += f"          <joint type=\"free\"/> \n"
    file_str += f"          <geom type=\"mesh\" mesh=\"task_board_{i}\" mass=\"0.2\"/> \n"
    file_str += f"      </body>\n"

file_str += "   </worldbody>\n"
file_str += "</mujocoinclude>"


with open("all_task_board.xml", "w") as text_file:
# with open("objects/rgmc_practice_task_board/all_task_board.xml", "w") as text_file:
    text_file.write(file_str)