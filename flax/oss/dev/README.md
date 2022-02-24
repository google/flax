We provide a docker setup that can be used with Visual Studio Code (VSCode).
The reason for using Docker is to provide a isolated development environment.
This should make setup easier and avoid issues caused by minor differences in
the environments used by contributors and maintainers.


# How to setup a dev environment using VSCode and Docker

1. Install Visual Studio Code and Docker.
2. Install the `Remote - Containers` extension within VSCode.
3. Copy the `.devcontainer` folder to the root of your workspace folder.
4. Make sure the source code of flax is checked out in `flax/`.
5. Re-open the folder workspace using the remote containers extension.
   VSCode should recommend this action in a popup. Alternatively,
   use the green button in the bottom left container to control the
   remote extension.