#Requires -Version 3.0

function Generate-DetailedFileList-Filtered-V2 {
    param (
        [string]$Path = ".",
        [string]$OutputFile = "FilteredFileList_V2.txt"
    )

    $RootPath = Resolve-Path $Path | Select-Object -ExpandProperty Path
    Write-Host "Analyzing folder structure for: $RootPath"
    Write-Host "Excluding .git and .vs folders and their subfolders."
    Write-Host "Saving output to: $OutputFile"

    # Define exclusion patterns using regex for the full path check
    $exclusionPatterns = "\.git\\", "\.vs\\"

    # Get items recursively and apply filter using Where-Object
    $allItems = Get-ChildItem -Path $RootPath -Recurse -Force

    # Filter out items where the FullName matches any of the exclusion patterns
    $filteredItems = $allItems | Where-Object {
        $include = $true
        foreach ($pattern in $exclusionPatterns) {
            if ($_.FullName -match $pattern) {
                $include = $false
                break
            }
        }
        $include
    }

    # Process the remaining items for formatting
    $results = $filteredItems | Select-Object FullName, PSIsContainer, CreationTime, LastWriteTime | ForEach-Object {
        [PSCustomObject]@{
            'Type'          = If ($_.PSIsContainer) {'Folder'} Else {'File'}
            'FullPath'      = $_.FullName
            'Created'       = $_.CreationTime.ToString("yyyy-MM-dd HH:mm:ss")
            'Modified'      = $_.LastWriteTime.ToString("yyyy-MM-dd HH:mm:ss")
        }
    }

    # Format the results into a clean table for the console
    $results | Format-Table -AutoSize

    # Save the results in a clear, structured list format to the output file
    $results | Format-List | Out-File -FilePath $OutputFile -Encoding UTF8

    Write-Host "Process complete."
}

# --- How to run the script ---
# Example for your specific path provided in the error message:
Generate-DetailedFileList-Filtered-V2 -Path "D:\Visual Studio 2022\ensemble_compression_publication_new_gpt"