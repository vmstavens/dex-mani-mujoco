for file in *.STL; do
    meshlabserver -i "$file" -o "${file%.STL}.obj"
done
