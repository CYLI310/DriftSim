-- DriftSim.app: double-click to start the dataset GUI server in the background and open it in
-- the browser. When it is already running, asks whether to open it or stop it.
-- Built by make_app.sh, which stores the repository path in Contents/Resources/repo_path.
-- All the real work is done by scripts/driftsim-gui.sh.

on run
	set launcher to my launcherPath()
	if launcher is "" then
		display dialog "DriftSim could not find its repository." & return & return & ¬
			"Rebuild the app with scripts/mac/make_app.sh in the DriftSim folder." ¬
			buttons {"OK"} default button "OK" with title "DriftSim" with icon stop
		return
	end if

	set serverURL to ""
	try
		set serverURL to do shell script quoted form of launcher & " --status"
	end try

	if serverURL is not "" then
		set choice to button returned of (display dialog "DriftSim is running at " & serverURL & return & return & ¬
			"Stopping it cancels a run in progress (files already written are kept)." ¬
			buttons {"Stop server", "Cancel", "Open"} default button "Open" cancel button "Cancel" ¬
			with title "DriftSim" with icon note)
		if choice is "Open" then
			open location serverURL
		else if choice is "Stop server" then
			try
				do shell script quoted form of launcher & " --stop"
				display notification "The server has stopped." with title "DriftSim"
			on error errMsg
				display dialog "Could not stop DriftSim:" & return & return & errMsg ¬
					buttons {"OK"} default button "OK" with title "DriftSim" with icon caution
			end try
		end if
	else
		display notification "Starting the dataset GUI..." with title "DriftSim"
		try
			-- prints the address and opens the browser once the server answers
			do shell script quoted form of launcher & " --background"
		on error errMsg
			set logFile to POSIX path of (path to home folder) & "Library/Logs/DriftSim/server.log"
			set choice to button returned of (display dialog "DriftSim could not start:" & return & return & errMsg ¬
				buttons {"Open log", "OK"} default button "OK" with title "DriftSim" with icon stop)
			if choice is "Open log" then do shell script "open " & quoted form of logFile & " || true"
		end try
	end if
end run

-- scripts/driftsim-gui.sh of the repository this app was built from, or "" if it is gone.
on launcherPath()
	set appPath to POSIX path of (path to me)
	set candidates to {}
	try
		set end of candidates to do shell script "cat " & quoted form of (appPath & "Contents/Resources/repo_path")
	end try
	-- fallback: the app sits in the repository folder
	set end of candidates to do shell script "dirname " & quoted form of appPath
	repeat with repo in candidates
		set p to (repo as text) & "/scripts/driftsim-gui.sh"
		try
			do shell script "test -x " & quoted form of p
			return p
		end try
	end repeat
	return ""
end launcherPath
