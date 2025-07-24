#!/bin/bash

if [ -n "$(pip freeze)" ]; then 
    echo "pip freeze returned installed packages. I advise running this script in a clean environment."

    read -p "Do you want to continue anyways? (y/n): " choice
    case "$choice" in
    y|Y )
        echo "Continuing..."
        ;;
    n|N )
        echo "Exiting script."
        exit 0
        ;;
    * )
        echo "Invalid input. Exiting."
        exit 1
        ;;
esac
fi

# Loop until the script runs successfully
while true; do
    # Run the Python script and capture the output and error
    output=$(python "$1" 2>&1)
    echo "$output"

    # Check if a ModuleNotFoundError occurred
    if echo "$output" | grep -q "ModuleNotFoundError"; then
        # Extract the missing module name
        module=$(echo "$output" | grep "ModuleNotFoundError" | sed -n "s/.*No module named '\([^']*\)'.*/\1/p")
        echo "Missing module: $module"

        while true; do
            echo "Trying to install '$module' with pip..."
            pip install "$module"
            
            # Check if pip succeeded
            if [ $? -eq 0 ]; then
                echo "Installed $module successfully."
                break
            else
                echo "Pip failed to install '$module'."
                read -p "Enter the correct pip package name for module '$module': " package
                pip install "$package"

                # Optionally update module name to prevent repeated failure
                if [ $? -eq 0 ]; then
                    echo "Installed $package successfully."
                    break
                else
                    echo "Still failed. Try again..."
                fi
            fi
        done
    else
        echo "Script ran without ModuleNotFoundError."
        pip freeze > guessed_requirements.txt
        break
    fi
done