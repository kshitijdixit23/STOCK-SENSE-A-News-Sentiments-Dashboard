import React from "react";
function Profile(){
  return(
    <img
          src="https://react.dev/images/docs/scientists/MK3eW3As.jpg"
          alt="Debojyoti Dey"
    />
  )
}
export default function Gallary(){
  return(
    <section>
      <h1>Debojyoti Dey's ancient biwi.</h1>
      <Profile/>
      <Profile/>
      <Profile/>
    </section>
  )
}